"""技术面数据整合：K线 + 成交量分析"""
from typing import Dict, Optional, Tuple
import logging

import pandas as pd

try:
    from gsuid_core.logger import logger
except ImportError:
    logger = logging.getLogger("SayuStock")  # type: ignore

try:
    from ..utils.constant import ErroText
    from ..utils.load_data import get_full_security_code
    from ..utils.stock.request import get_gg
    from ..utils.stock.request_utils import get_code_id
    _HAS_FRAMEWORK = True
except ImportError:
    _HAS_FRAMEWORK = False
    ErroText = {"notStock": "未找到该股票"}  # type: ignore


def _parse_kline_df(raw_data: Dict) -> Optional[pd.DataFrame]:
    if not raw_data.get("data") or not raw_data["data"].get("klines"):
        return None
    headers = [
        "date", "open", "close", "high", "low",
        "volume", "amount", "amplitude", "chg_percent",
        "chg_amount", "turnover_rate",
    ]
    rows = [line.split(",") for line in raw_data["data"]["klines"]]
    df = pd.DataFrame(rows, columns=headers)
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build_tech_summary(df: pd.DataFrame, stock_name: str, current_price: float) -> str:
    """根据 DataFrame 生成技术面文字摘要，供 LLM 使用。"""
    close = df["close"]
    volume = df["volume"]

    period_high = close.max()
    period_low = close.min()
    price_range = period_high - period_low
    position_pct = (current_price - period_low) / price_range * 100 if price_range > 0 else 50.0

    ma5 = close.tail(5).mean()    # ≈ 1.25 交易日
    ma20 = close.tail(20).mean()  # ≈ 5 交易日
    ma5_trend = "上方" if current_price > ma5 else "下方"
    ma20_trend = "上方" if current_price > ma20 else "下方"

    vol5 = volume.tail(5).mean()
    vol20 = volume.tail(20).mean()
    vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0
    if vol_ratio > 1.2:
        vol_desc = f"近5根均量是近20根均量的{vol_ratio:.1f}倍，明显放量"
    elif vol_ratio < 0.8:
        vol_desc = f"近5根均量是近20根均量的{vol_ratio:.1f}倍，明显缩量"
    else:
        vol_desc = f"近5根与近20根均量相近（比值{vol_ratio:.2f}），量能平稳"

    chg_5h = (current_price - close.iloc[-5]) / close.iloc[-5] * 100 if len(close) >= 5 else 0.0
    chg_all = (current_price - close.iloc[0]) / close.iloc[0] * 100 if len(close) >= 2 else 0.0

    summary = (
        f"【{stock_name}】技术面摘要（近5个交易日日K）：\n"
        f"- 当前价：{current_price:.2f}元\n"
        f"- 本周价格区间：{period_low:.2f} ~ {period_high:.2f}，当前处于区间{position_pct:.1f}%分位\n"
        f"- MA5={ma5:.2f}，MA20={ma20:.2f}，当前价在MA5{ma5_trend}、MA20{ma20_trend}\n"
        f"- 本周涨跌：{chg_all:+.2f}%，昨日涨跌：{chg_5h:+.2f}%\n"
        f"- 成交量：{vol_desc}"
    )
    return summary


def _get_market_name(sec_id: str) -> str:
    """根据 secid（如 1.600519 / 116.00700）判断所属大盘。"""
    parts = sec_id.split(".")
    if len(parts) != 2:
        return ""
    market, code = parts[0], parts[1]
    if market == "1":
        if code.startswith("688"):
            return "科创板"
        return "沪市主板"
    elif market == "0":
        if code.startswith("300") or code.startswith("301"):
            return "创业板"
        if code.startswith("83") or code.startswith("87"):
            return "北交所"
        return "深市主板"
    elif market in ("105", "106", "107"):
        return "美股"
    elif market == "116":
        return "港股"
    return ""


async def get_stock_tech_data(
    code: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[pd.DataFrame], Optional[str], float, float, float]:
    """
    获取个股技术面数据。

    Returns:
        (stock_name, stock_code, market_name, df, tech_summary, current_price, chg_amount, chg_pct)
        失败时 stock_name=None
    """
    sec_id_data = await get_code_id(code)
    if sec_id_data is None:
        return None, None, None, None, ErroText["notStock"], 0.0, 0.0, 0.0

    sec_id = get_full_security_code(sec_id_data[0])
    if sec_id is None:
        return None, None, None, None, ErroText["notStock"], 0.0, 0.0, 0.0

    # 从 secid（如 "1.600519"）提取代码
    stock_code: str = sec_id.split(".")[-1] if "." in sec_id else sec_id
    market_name = _get_market_name(sec_id)

    # 获取基础行情（含股票名、当前价、涨跌）
    basic = await get_gg(sec_id, "single-stock")
    if isinstance(basic, str):
        return None, None, None, None, basic, 0.0, 0.0, 0.0

    # f58 格式为 "贵州茅台 (A)"，去掉括号后缀
    stock_name: str = basic["data"].get("f58", code).split("(")[0].strip()
    current_price: float = float(basic["data"].get("f43", 0) or 0)

    raw_chg_amt = basic["data"].get("f169", 0)
    raw_chg_pct = basic["data"].get("f170", 0)
    chg_amount: float = float(raw_chg_amt) if not isinstance(raw_chg_amt, str) else 0.0
    chg_pct: float = float(raw_chg_pct) if not isinstance(raw_chg_pct, str) else 0.0

    # 获取日K（近60个交易日，用于图表展示）
    kline_raw = await get_gg(sec_id, "single-stock-kline-101")
    if isinstance(kline_raw, str):
        return None, None, None, None, kline_raw, 0.0, 0.0, 0.0

    df = _parse_kline_df(kline_raw)
    if df is None or df.empty:
        return None, None, None, None, "无有效K线数据", 0.0, 0.0, 0.0

    # 图表展示最近60根（约60个交易日）
    df = df.tail(60).reset_index(drop=True)

    # 技术摘要只取最近5根（近一周），供 Kimi 分析
    tech_summary = _build_tech_summary(df.tail(5), stock_name, current_price)
    logger.info(f"[SayuStock] 技术面摘要生成完成: {stock_name}({stock_code}) [{market_name}]")
    return stock_name, stock_code, market_name, df, tech_summary, current_price, chg_amount, chg_pct
