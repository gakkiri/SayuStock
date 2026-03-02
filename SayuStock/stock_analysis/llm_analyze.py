"""调用 Kimi（含 $web_search 工具）对股票进行综合分析"""
import json
import logging
import asyncio
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple
from pathlib import Path

from openai import AsyncOpenAI
from aiohttp import ClientSession, TCPConnector, ClientTimeout

try:
    from gsuid_core.logger import logger
except ImportError:
    logger = logging.getLogger("SayuStock")  # type: ignore

def _load_api_key() -> str:
    key_file = Path.home() / "kimi_api"
    try:
        return key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.error(f"[SayuStock] Kimi API key 文件不存在: {key_file}")
        return ""

KIMI_API_KEY = _load_api_key()
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL = "kimi-k2-turbo-preview"  # "kimi-k2.5"
MAX_RETRIES = 5
RETRY_DELAY = 5  # 秒
REQUEST_TIMEOUT = 300  # 单轮请求超时（秒）
MAX_ROUNDS = 8  # 单次调用最多循环轮数

TAVILY_BASE_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT = 20  # 秒

def _load_tavily_key() -> str:
    key_file = Path.home() / "tavily_api"
    try:
        return key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"[SayuStock] Tavily API key 文件不存在: {key_file}，跳过宏观预取")
        return ""

TAVILY_API_KEY = _load_tavily_key()

# 近3天宏观舆情搜索词：覆盖 A股政策、中国宏观、外部市场三个维度
_MACRO_QUERIES = [
    "A股 政策 监管 资金 市场情绪",
    "中国宏观经济 货币政策 央行 利率",
    "美联储 美股 美元 国际市场 A股影响",
]

async def _tavily_search(session: ClientSession, query: str) -> List[Dict]:
    """单条 Tavily 新闻搜索，返回结果列表，失败返回空列表。"""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "topic": "news",
        "days": 3,
        "max_results": 4,
    }
    try:
        async with session.post(
            TAVILY_BASE_URL,
            json=payload,
            timeout=ClientTimeout(total=TAVILY_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"[SayuStock] Tavily 搜索失败 status={resp.status} query={query!r}")
                return []
            data = await resp.json()
            return data.get("results", [])
    except Exception as e:
        logger.warning(f"[SayuStock] Tavily 搜索异常 query={query!r}: {e}")
        return []


async def fetch_tavily_macro_news() -> str:
    """
    并发拉取近3天与 A股强相关的宏观舆情，返回格式化文本供 Kimi 参考。
    未配置 Tavily key 或全部请求失败时返回空字符串。
    """
    if not TAVILY_API_KEY:
        return ""

    async with ClientSession(connector=TCPConnector(ssl=True)) as session:
        results_list = await asyncio.gather(
            *[_tavily_search(session, q) for q in _MACRO_QUERIES],
            return_exceptions=True,
        )

    seen_urls: set = set()
    items: List[str] = []
    for results in results_list:
        if isinstance(results, Exception):
            continue
        for r in results:
            url = r.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = r.get("title", "").strip()
            content = r.get("content", "").strip()[:150]
            pub = (r.get("published_date") or "")[:10]
            date_tag = f"({pub})" if pub else ""
            items.append(f"- {title}{date_tag}: {content}")

    if not items:
        logger.info("[SayuStock] Tavily 宏观预取结果为空")
        return ""

    logger.info(f"[SayuStock] Tavily 宏观预取 {len(items)} 条")
    return "\n".join(items[:10])


def _build_system_prompt() -> str:
    today = date.today().strftime('%Y-%m-%d')
    return (
        f"你是一位专业的 A 股分析师，今天是 {today}。"
        f"用户消息中会附带【近3天宏观舆情】预取材料，请直接采用，无需重复搜索相同内容。"
        f"搜索时请只关注时效性强的近期内容，优先引用距今一周以内的信源，过时信息请忽略。"
        f"搜索完成后，【直接】以 JSON 格式输出分析结论，不得输出任何其他文字。\n"
        f"JSON 必须严格包含以下字段：\n"
        f'{{"verdict": "多空判断，看多/看空/中性，不超过20字",'
        f' "reason": "核心理由，分3~4条展开：①技术面（均线/MACD/RSI/KDJ/布林带中最关键的信号）②个股舆情（近期公告/新闻要点）③宏观面（对本股影响最直接的宏观因素）④量价配合（成交量与价格走势的关系），每条1~2句，关键词用**加粗**，总计150~300字",'
        f' "risk": "风险提示，一条，不超过40字",'
        f' "news_summary": "对近期新闻公告的一句话概括，不超过60字",'
        f' "sources": ["来源名称(YYYY-MM-DD): 内容摘要（最多5条）"]}}'
    )


def search_impl(arguments: Dict[str, Any]) -> Any:
    """与 kimi_official.py 保持一致：直接返回 arguments，由 Kimi 服务端完成实际搜索。"""
    logger.info(f"[SayuStock] AI 搜索: {arguments.get('query', '')}")
    return arguments


async def _kimi_call(
    client: AsyncOpenAI,
    stock_name: str,
    tech_summary: str,
    macro_news: str,
) -> Tuple[str, str, str, str, List[str]]:
    """单次 Kimi 调用，出错时抛异常由外层重试。"""
    macro_section = (
        f"\n\n【近3天宏观舆情（Tavily 预取，请直接引用）】\n{macro_news}"
        if macro_news else ""
    )
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {
            "role": "user",
            "content": (
                f"{tech_summary}{macro_section}\n\n"
                f"今天是 {date.today().strftime('%Y-%m-%d')}，"
                f"请针对个股补充搜索：\n"
                f"1. 【{stock_name}】近一个月（{(date.today() - timedelta(days=180)).strftime('%Y-%m-%d')} 至今）的新闻、公告、市场舆情；\n"
                f"2. 近期可能影响 A 股整体走势的重大宏观事件（宏观方向已在上方预取材料中，可侧重未覆盖部分）。\n"
                f"结合以上技术面数据及搜索结果，按要求格式给出分析结论。"
            ),
        },
    ]

    tools = [
        {
            "type": "builtin_function",
            "function": {"name": "$web_search"},
        }
    ]

    finish_reason = None
    choice = None
    rounds = 0
    while finish_reason is None or finish_reason == "tool_calls":
        rounds += 1
        if rounds > MAX_ROUNDS:
            raise ValueError(f"Kimi 循环超过 {MAX_ROUNDS} 轮，强制终止")
        logger.info(f"[SayuStock] Kimi 第{rounds}轮请求（finish_reason={finish_reason}）")
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=KIMI_MODEL,
                    messages=messages,
                    extra_body={"thinking": {"type": "disabled"}},
                    tools=tools,
                    response_format={"type": "json_object"},
                ),
                timeout=REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise ValueError(f"Kimi 第{rounds}轮请求超时（>{REQUEST_TIMEOUT}s）")
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        logger.info(f"[SayuStock] Kimi 响应 finish_reason={finish_reason}")
        if finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(tool_call.function.arguments)
                if tool_call_name == "$web_search":
                    tool_result = search_impl(tool_call_arguments)
                else:
                    tool_result = f"Error: unable to find tool by name '{tool_call_name}'"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result),
                })

    if choice is None or not choice.message.content:
        raise ValueError(f"Kimi 返回内容为空（finish_reason={finish_reason}）")

    raw = choice.message.content
    logger.info(f"[SayuStock] AI 原始回答:\n{raw}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Kimi 返回内容非法 JSON: {e}，原始内容: {raw[:200]}")

    verdict = str(data.get("verdict", "中性")).strip()
    if len(verdict) > 30:
        raise ValueError(f"verdict 格式异常（{len(verdict)}字），原始内容: {raw[:200]}")

    reason       = str(data.get("reason", "暂无分析")).strip()
    risk         = str(data.get("risk", "注意市场风险")).strip()
    news_summary = str(data.get("news_summary", "")).strip()
    sources_raw  = data.get("sources", [])
    if isinstance(sources_raw, list):
        sources = [s.strip() for s in sources_raw if isinstance(s, str) and s.strip()]
    else:
        sources = [str(sources_raw).strip()] if sources_raw else []

    logger.info(f"[SayuStock] 分析完成: {verdict}, 信源 {len(sources)} 条")
    return verdict, reason, risk, news_summary, sources


async def kimi_analyze(
    stock_name: str,
    tech_summary: str,
) -> Tuple[str, str, str, str, List[str]]:
    """调用 Kimi 分析股票，失败自动重试，返回 (判断, 理由, 风险, 舆情摘要, 信源列表)"""
    client = AsyncOpenAI(base_url=KIMI_BASE_URL, api_key=KIMI_API_KEY)
    last_err: Exception = RuntimeError("未知错误")

    # Tavily 宏观预取只做一次，失败不影响主流程
    macro_news = await fetch_tavily_macro_news()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[SayuStock] Kimi 分析第{attempt}/{MAX_RETRIES}次尝试")
            return await _kimi_call(client, stock_name, tech_summary, macro_news)
        except Exception as e:
            last_err = e
            logger.warning(f"[SayuStock] Kimi 第{attempt}次失败: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"[SayuStock] {RETRY_DELAY}秒后重试...")
                await asyncio.sleep(RETRY_DELAY)

    logger.error(f"[SayuStock] Kimi 全部{MAX_RETRIES}次重试均失败，最后错误: {last_err}")
    raise last_err
