"""调用 Kimi（含 $web_search 工具）对股票进行综合分析"""
import json
import re
import os
import logging
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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
MACRO_LOOKBACK_DAYS = 3
NEWS_LOOKBACK_DAYS = 30
MAX_DISPLAY_SOURCES = 6


def _load_tavily_key() -> str:
    env_key = os.getenv("TAVILY_API_KEYS", "").strip()
    if env_key:
        return env_key
    key_file = Path.home() / "tavily_api"
    try:
        return key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"[SayuStock] Tavily API key 文件不存在: {key_file}，跳过前置搜索")
        return ""


TAVILY_API_KEY = _load_tavily_key()

# 近3天宏观舆情搜索词：覆盖 A股政策、中国宏观、外部市场三个维度
_MACRO_QUERIES = [
    "A股 政策 监管 资金 市场情绪",
    "中国宏观经济 货币政策 央行 利率",
    "美联储 美股 美元 国际市场 A股影响",
]

_SOURCE_PATTERN = re.compile(
    r"^\s*(?P<source>.+?)\((?P<date>\d{4}-\d{2}-\d{2})\)\s*[:：]\s*(?P<content>.+?)\s*$"
)


@dataclass(frozen=True)
class PrefetchedNewsItem:
    title: str
    summary: str
    published_date: str
    url: str
    source_name: str
    category: str


def _normalize_date_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""

    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)

    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return ""
    return parsed.date().strftime("%Y-%m-%d")


def _parse_date(value: Any) -> Optional[date]:
    text = _normalize_date_text(value)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _is_future_date(value: Any, today: Optional[date] = None) -> bool:
    published = _parse_date(value)
    if published is None:
        return False
    return published > (today or date.today())


def _build_prefetch_requests(stock_name: str) -> List[Dict[str, Any]]:
    requests = [
        {
            "category": "macro",
            "query": query,
            "days": MACRO_LOOKBACK_DAYS,
            "max_results": 4,
        }
        for query in _MACRO_QUERIES
    ]
    stock_query = stock_name.strip()
    if stock_query:
        requests.append(
            {
                "category": "stock",
                "query": f"{stock_query} 股票 时事热点",
                "days": NEWS_LOOKBACK_DAYS,
                "max_results": 6,
            }
        )
    return requests


def _extract_source_name(raw: Dict[str, Any]) -> str:
    source_name = str(raw.get("source") or raw.get("site_name") or "").strip()
    if source_name:
        return source_name
    url = str(raw.get("url") or "").strip()
    if not url:
        return "未知来源"
    hostname = urlparse(url).netloc.lower().strip()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname or "未知来源"


def _normalize_prefetched_item(
    raw: Dict[str, Any],
    category: str,
) -> Optional[PrefetchedNewsItem]:
    title = str(raw.get("title") or "").strip()
    summary = str(raw.get("content") or "").strip()
    published_date = _normalize_date_text(raw.get("published_date"))
    url = str(raw.get("url") or "").strip()
    source_name = _extract_source_name(raw)

    if not title and not summary:
        return None

    if _is_future_date(published_date):
        return None

    if not title:
        title = summary[:60]
    if not summary:
        summary = title

    return PrefetchedNewsItem(
        title=title,
        summary=summary[:180],
        published_date=published_date,
        url=url,
        source_name=source_name,
        category=category,
    )


async def _tavily_search(
    session: ClientSession,
    query: str,
    days: int,
    max_results: int,
) -> List[Dict]:
    """单条 Tavily 新闻搜索，返回结果列表，失败返回空列表。"""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "topic": "news",
        "days": days,
        "max_results": max_results,
    }
    try:
        async with session.post(
            TAVILY_BASE_URL,
            json=payload,
            timeout=ClientTimeout(total=TAVILY_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                logger.warning(
                    f"[SayuStock] Tavily 搜索失败 status={resp.status} query={query!r}"
                )
                return []
            data = await resp.json()
            return data.get("results", [])
    except Exception as e:
        logger.warning(f"[SayuStock] Tavily 搜索异常 query={query!r}: {e}")
        return []


def _prefetch_sort_key(item: PrefetchedNewsItem) -> date:
    return _parse_date(item.published_date) or date.min


async def fetch_tavily_prefetched_news(stock_name: str) -> List[PrefetchedNewsItem]:
    """
    前置搜索：宏观 + 个股。
    返回结构化结果，供 Kimi 参考和最终展示信源合并使用。
    """
    if not TAVILY_API_KEY:
        return []

    requests = _build_prefetch_requests(stock_name)
    if not requests:
        return []

    async with ClientSession(connector=TCPConnector(ssl=True)) as session:
        results_list = await asyncio.gather(
            *[
                _tavily_search(
                    session,
                    req["query"],
                    req["days"],
                    req["max_results"],
                )
                for req in requests
            ],
            return_exceptions=True,
        )

    deduped: List[PrefetchedNewsItem] = []
    seen_keys = set()
    for req, results in zip(requests, results_list):
        if isinstance(results, Exception):
            continue
        for raw in results:
            item = _normalize_prefetched_item(raw, req["category"])
            if item is None:
                continue
            dedupe_key = item.url or "|".join(
                [item.source_name, item.published_date, item.title]
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped.append(item)

    deduped.sort(key=_prefetch_sort_key, reverse=True)
    logger.info(f"[SayuStock] Tavily 前置搜索共 {len(deduped)} 条")
    return deduped[:12]


def _format_prefetched_news_for_prompt(items: List[PrefetchedNewsItem]) -> str:
    if not items:
        return ""

    lines = []
    for idx, item in enumerate(items, start=1):
        category_name = "个股" if item.category == "stock" else "宏观"
        date_text = item.published_date or "无明确日期"
        lines.append(
            f"[{idx}][{category_name}][{date_text}][{item.source_name}] "
            f"{item.title}；摘要：{item.summary}"
        )
    return "【前置搜索舆情】\n" + "\n".join(lines)


def _build_system_prompt() -> str:
    today = date.today().strftime("%Y-%m-%d")
    return (
        f"你是一位专业的 A 股分析师，今天是 {today}。"
        f"用户消息中会附带【前置搜索舆情】材料，其中包含宏观与个股新闻，你必须优先引用【前置搜索舆情】。"
        f"在输出最终 JSON 之前，必须至少调用一次 $web_search 进行补充搜索或交叉验证。"
        f"个股新闻、公告、时事热点原则上只使用30天内且有明确日期的材料；若30天内完全没有任何信源，才可回退到更早但仍相关的有日期材料。"
        f"严禁引用晚于今天的未来日期信源。"
        f"前置搜索舆情可以直接写入 sources 字段。"
        f"搜索完成后，【直接】以 JSON 格式输出分析结论，不得输出任何其他文字。\n"
        f"JSON 必须严格包含以下字段：\n"
        f'{{"verdict": "多空判断，看多/看空/中性，不超过20字",'
        f' "reason": "核心理由，分3~4条展开：①技术面（以机器人提供的K线/指标摘要为准）②个股舆情（优先引用前置搜索与近30天公告/热点）③宏观面（优先引用前置搜索中的宏观材料）④量价配合（成交量与价格走势的关系），每条1~2句，关键词用**加粗**，总计150~300字",'
        f' "risk": "风险提示，一条，不超过40字",'
        f' "news_summary": "对近期新闻公告的一句话概括，不超过60字",'
        f' "sources": ["来源名称(YYYY-MM-DD): 内容摘要（最多5条，优先30天内）"]}}'
    )


def _build_user_prompt(
    stock_name: str,
    stock_code: str,
    market_name: str,
    tech_summary: str,
    prefetched_news_text: str,
) -> str:
    today = date.today().strftime("%Y-%m-%d")
    stock_hint = f"{stock_name}（{stock_code}，{market_name or '市场未识别'}）"
    prefetched_section = (
        f"{prefetched_news_text}\n\n"
        if prefetched_news_text
        else "【前置搜索舆情】\n暂无可用前置搜索结果，请通过搜索补齐。\n\n"
    )
    return (
        f"{tech_summary}\n\n"
        f"今天是 {today}。以上技术面/K线摘要来自机器人侧提供的近60个交易日数据，请直接基于该摘要分析，不要擅自改写技术面时间范围。\n\n"
        f"{prefetched_section}"
        f"请针对 {stock_hint} 执行分析：\n"
        f"1. 必须至少调用一次 $web_search，对该股近30天的新闻、公告、时事热点进行补充搜索或交叉验证；\n"
        f"2. 如前置搜索中的宏观材料不足，可再搜索近期影响 A 股整体走势的重大宏观事件；\n"
        f"3. 最终 sources 字段允许同时引用前置搜索舆情和 $web_search 搜到的结果，但应优先保留近30天内且有明确日期的信源。\n"
        f"结合以上技术面与舆情材料，按要求输出 JSON。"
    )


def search_impl(arguments: Dict[str, Any]) -> Any:
    """与 kimi_official.py 保持一致：直接返回 arguments，由 Kimi 服务端完成实际搜索。"""
    logger.info(f"[SayuStock] AI 搜索: {arguments.get('query', '')}")
    return arguments


def _format_prefetched_display_source(item: PrefetchedNewsItem) -> Optional[str]:
    published = _normalize_date_text(item.published_date)
    if not published:
        return None
    body = item.title
    if item.summary and item.summary not in item.title:
        body = f"{item.title}；{item.summary}"
    return f"{item.source_name}({published}): {body}"


def _parse_kimi_source(source_text: str) -> Optional[Tuple[date, str]]:
    match = _SOURCE_PATTERN.match(str(source_text).strip())
    if not match:
        return None
    published = _parse_date(match.group("date"))
    if published is None:
        return None
    if published > date.today():
        return None
    source_name = match.group("source").strip()
    content = match.group("content").strip()
    if not source_name or not content:
        return None
    text = f"{source_name}({published.strftime('%Y-%m-%d')}): {content}"
    return published, text


def _merge_display_sources(
    prefetched_items: List[PrefetchedNewsItem],
    kimi_sources: List[str],
    limit: int = MAX_DISPLAY_SOURCES,
) -> List[str]:
    today = date.today()
    cutoff = date.today() - timedelta(days=NEWS_LOOKBACK_DAYS)
    recent_candidates: List[Tuple[date, str]] = []
    older_candidates: List[Tuple[date, str]] = []
    seen_texts = set()

    for item in prefetched_items:
        published = _parse_date(item.published_date)
        formatted = _format_prefetched_display_source(item)
        if published is None or formatted is None:
            continue
        if published > today:
            continue
        if formatted in seen_texts:
            continue
        seen_texts.add(formatted)
        bucket = recent_candidates if published >= cutoff else older_candidates
        bucket.append((published, formatted))

    for source_text in kimi_sources:
        parsed = _parse_kimi_source(source_text)
        if parsed is None:
            continue
        published, formatted = parsed
        if formatted in seen_texts:
            continue
        seen_texts.add(formatted)
        bucket = recent_candidates if published >= cutoff else older_candidates
        bucket.append((published, formatted))

    selected = recent_candidates if recent_candidates else older_candidates
    selected.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in selected[:limit]]


async def _kimi_call(
    client: AsyncOpenAI,
    stock_name: str,
    stock_code: str,
    market_name: str,
    tech_summary: str,
    prefetched_items: List[PrefetchedNewsItem],
) -> Tuple[str, str, str, str, List[str]]:
    """单次 Kimi 调用，出错时抛异常由外层重试。"""
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {
            "role": "user",
            "content": _build_user_prompt(
                stock_name,
                stock_code,
                market_name,
                tech_summary,
                _format_prefetched_news_for_prompt(prefetched_items),
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
    used_web_search = False
    forced_search_reminder_sent = False

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
                    used_web_search = True
                    tool_result = search_impl(tool_call_arguments)
                else:
                    tool_result = f"Error: unable to find tool by name '{tool_call_name}'"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call_name,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            continue

        if not used_web_search:
            if forced_search_reminder_sent:
                raise ValueError("Kimi 未按要求执行搜索")
            forced_search_reminder_sent = True
            finish_reason = None
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "你还没有调用 $web_search。必须至少调用一次 $web_search，"
                        "优先搜索该股近30天新闻/公告/时事热点，再继续输出最终 JSON。"
                    ),
                }
            )

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

    reason = str(data.get("reason", "暂无分析")).strip()
    risk = str(data.get("risk", "注意市场风险")).strip()
    news_summary = str(data.get("news_summary", "")).strip()
    sources_raw = data.get("sources", [])
    if isinstance(sources_raw, list):
        kimi_sources = [
            s.strip() for s in sources_raw if isinstance(s, str) and s.strip()
        ]
    else:
        kimi_sources = [str(sources_raw).strip()] if sources_raw else []

    sources = _merge_display_sources(prefetched_items, kimi_sources)
    logger.info(f"[SayuStock] 分析完成: {verdict}, 信源 {len(sources)} 条")
    return verdict, reason, risk, news_summary, sources


async def kimi_analyze(
    stock_name: str,
    stock_code: str,
    market_name: str,
    tech_summary: str,
) -> Tuple[str, str, str, str, List[str]]:
    """调用 Kimi 分析股票，失败自动重试，返回 (判断, 理由, 风险, 舆情摘要, 信源列表)"""
    client = AsyncOpenAI(base_url=KIMI_BASE_URL, api_key=KIMI_API_KEY)
    last_err: Exception = RuntimeError("未知错误")

    prefetched_items = await fetch_tavily_prefetched_news(stock_name)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[SayuStock] Kimi 分析第{attempt}/{MAX_RETRIES}次尝试")
            return await _kimi_call(
                client,
                stock_name,
                stock_code,
                market_name,
                tech_summary,
                prefetched_items,
            )
        except Exception as e:
            last_err = e
            logger.warning(f"[SayuStock] Kimi 第{attempt}次失败: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"[SayuStock] {RETRY_DELAY}秒后重试...")
                await asyncio.sleep(RETRY_DELAY)

    logger.error(
        f"[SayuStock] Kimi 全部{MAX_RETRIES}次重试均失败，最后错误: {last_err}"
    )
    raise last_err
