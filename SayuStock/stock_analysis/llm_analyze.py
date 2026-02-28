"""调用 Kimi（含 $web_search 工具）对股票进行综合分析"""
import re
import json
import logging
import asyncio
from typing import Any, Dict, List, Tuple
from pathlib import Path

from openai import AsyncOpenAI

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
KIMI_MODEL = "kimi-k2.5"
MAX_RETRIES = 5
RETRY_DELAY = 5  # 秒

SYSTEM_PROMPT = """你是一位专业的 A 股分析师，擅长结合技术面与舆情做出研判。
搜索完成后，输出格式严格如下，用 |sep| 分隔五个部分，不要有其他任何多余内容：
多空判断（看多/看空/中性，一句话，不超过20字）|sep|核心理由（技术面一条+舆情一条，合计不超过100字，可用markdown加粗关键词）|sep|风险提示（一条，不超过40字）|sep|舆情摘要（对近期新闻公告的一句话概括，不超过60字）|sep|信源列表（每条格式严格为"来源名称(MM-DD): 内容摘要"，多条之间用 ;; 分隔，不超过5条，例如：新浪财经(02-28): 茅台发布年报营收同比增长15%;;东方财富公告(02-27): 公司拟回购股票不超过5亿元;;雪球(02-26): 机构大幅增持，市场情绪偏多）"""


def search_impl(arguments: Dict[str, Any]) -> Any:
    """与 kimi_official.py 保持一致：直接返回 arguments，由 Kimi 服务端完成实际搜索。"""
    logger.info(f"[SayuStock] AI 搜索: {arguments.get('query', '')}")
    return arguments


def _extract_sources(content: str) -> Tuple[str, List[str]]:
    """
    从 Kimi 最终回答中提取引用文献。
    Kimi 搜索后会在回答末尾附加 [^N]: 标题 URL 格式的参考文献块。
    返回 (去掉文献块的正文, 文献标题列表)
    """
    sources = []
    ref_pattern = re.compile(r'^\[\^(\d+)\]:\s*(.+?)(?:\s+(https?://\S+))?$', re.MULTILINE)
    for m in ref_pattern.finditer(content):
        title = m.group(2).strip()
        url = (m.group(3) or "").strip()
        sources.append(f"{title}  {url}".strip() if url else title)

    clean = re.sub(r'\n*\[\^\d+\]:.*', '', content, flags=re.DOTALL).strip()
    return clean, sources


async def _kimi_call(
    client: AsyncOpenAI,
    stock_name: str,
    tech_summary: str,
) -> Tuple[str, str, str, str, List[str]]:
    """单次 Kimi 调用，出错时抛异常由外层重试。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{tech_summary}\n\n"
                f"请搜索【{stock_name}】近期新闻、公告、市场舆情，"
                f"结合以上技术面数据，按要求格式给出分析结论。"
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
        logger.info(f"[SayuStock] Kimi 第{rounds}轮请求（finish_reason={finish_reason}）")
        completion = await client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}},
            tools=tools,
        )
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

    parts = [p.strip() for p in raw.split("|sep|")]
    if len(parts) < 2:
        raise ValueError(f"Kimi 回答格式不符（parts={len(parts)}），原始内容: {raw[:200]}")

    verdict      = parts[0] if len(parts) > 0 else "中性"
    reason       = parts[1] if len(parts) > 1 else "暂无分析"
    risk         = parts[2] if len(parts) > 2 else "注意市场风险"
    news_summary = parts[3] if len(parts) > 3 else ""
    sources_raw  = parts[4] if len(parts) > 4 else ""
    # 优先按 ;; 分割，回退到换行/顿号/逗号
    sources = [s.strip() for s in sources_raw.split(';;') if s.strip()]
    if not sources:
        sources = [s.strip() for s in re.split(r'[\n、，,]', sources_raw) if s.strip()]

    logger.info(f"[SayuStock] 分析完成: {verdict}, 信源 {len(sources)} 条")
    return verdict, reason, risk, news_summary, sources


async def kimi_analyze(
    stock_name: str,
    tech_summary: str,
) -> Tuple[str, str, str, str, List[str]]:
    """调用 Kimi 分析股票，失败自动重试，返回 (判断, 理由, 风险, 舆情摘要, 信源列表)"""
    client = AsyncOpenAI(base_url=KIMI_BASE_URL, api_key=KIMI_API_KEY)
    last_err: Exception = RuntimeError("未知错误")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[SayuStock] Kimi 分析第{attempt}/{MAX_RETRIES}次尝试")
            return await _kimi_call(client, stock_name, tech_summary)
        except Exception as e:
            last_err = e
            logger.warning(f"[SayuStock] Kimi 第{attempt}次失败: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"[SayuStock] {RETRY_DELAY}秒后重试...")
                await asyncio.sleep(RETRY_DELAY)

    logger.error(f"[SayuStock] Kimi 全部{MAX_RETRIES}次重试均失败，最后错误: {last_err}")
    return "中性", f"AI分析失败（{last_err}）", "注意市场风险", "", []
