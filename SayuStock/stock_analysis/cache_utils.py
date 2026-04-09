import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable, Optional, Tuple

try:
    from gsuid_core.logger import logger
except ImportError:
    logger = logging.getLogger("SayuStock")  # type: ignore


ANALYSIS_IMAGE_CACHE_DIR = Path(__file__).parent / "image_cache"
ANALYSIS_IMAGE_CACHE_DIR.mkdir(exist_ok=True)
ANALYSIS_CACHE_TTL = timedelta(minutes=30)


def _sanitize_stock_code(stock_code: str) -> str:
    return "".join(ch for ch in stock_code if ch.isalnum()) or stock_code


def get_analysis_cache_file(
    stock_code: str,
    cache_dir: Path = ANALYSIS_IMAGE_CACHE_DIR,
) -> Path:
    return cache_dir / f"{_sanitize_stock_code(stock_code)}_analysis.png"


def load_cached_analysis_image(
    stock_code: str,
    cache_dir: Path = ANALYSIS_IMAGE_CACHE_DIR,
    now: Optional[datetime] = None,
) -> Optional[bytes]:
    cache_file = get_analysis_cache_file(stock_code, cache_dir=cache_dir)
    if not cache_file.exists():
        return None

    now = now or datetime.now()
    modified_at = datetime.fromtimestamp(cache_file.stat().st_mtime)
    if now - modified_at > ANALYSIS_CACHE_TTL:
        return None

    try:
        return cache_file.read_bytes()
    except OSError as e:
        logger.warning(f"[SayuStock] 读取分析缓存失败 {cache_file}: {e}")
        return None


def save_cached_analysis_image(
    stock_code: str,
    image_bytes: bytes,
    cache_dir: Path = ANALYSIS_IMAGE_CACHE_DIR,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = get_analysis_cache_file(stock_code, cache_dir=cache_dir)
    cache_file.write_bytes(image_bytes)
    return cache_file


async def resolve_cache_stock_code(
    code: str,
    resolver: Optional[Callable[[str], Awaitable[Optional[Tuple[str, str, str]]]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    if resolver is None:
        from ..utils.stock.request_utils import get_code_id

        resolver = get_code_id
    if normalizer is None:
        from ..utils.load_data import get_full_security_code

        normalizer = get_full_security_code

    sec_id_data = await resolver(code)
    if sec_id_data is None:
        return None

    try:
        sec_id = normalizer(sec_id_data[0])
    except Exception as e:
        logger.warning(f"[SayuStock] 解析分析缓存键失败 code={code}: {e}")
        return None

    if not sec_id:
        return None

    return sec_id.split(".")[-1] if "." in sec_id else sec_id


async def get_cached_analysis_image_for_query(
    code: str,
    cache_dir: Path = ANALYSIS_IMAGE_CACHE_DIR,
    now: Optional[datetime] = None,
    resolver: Optional[Callable[[str], Awaitable[Optional[Tuple[str, str, str]]]]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
) -> Optional[bytes]:
    stock_code = await resolve_cache_stock_code(
        code,
        resolver=resolver,
        normalizer=normalizer,
    )
    if not stock_code:
        return None
    return load_cached_analysis_image(
        stock_code,
        cache_dir=cache_dir,
        now=now,
    )
