"""技术面数据整合：K线 + 成交量分析"""
from typing import Any, Dict, Optional, Tuple
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


INVALID_NUM_TOKENS = {"", "-", "--", "null", "None", "nan", "NaN"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if text in INVALID_NUM_TOKENS:
        return default

    try:
        return float(text)
    except (TypeError, ValueError):
        return default


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
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    n      = len(df)

    # ── 1. 均线 ──────────────────────────────────────────────────
    ma5  = close.tail(5).mean()
    ma10 = close.tail(10).mean() if n >= 10 else None
    ma20 = close.tail(20).mean() if n >= 20 else None
    ma60 = close.tail(60).mean() if n >= 60 else None

    ma_parts = []
    for label, val in [("MA5", ma5), ("MA10", ma10), ("MA20", ma20), ("MA60", ma60)]:
        if val is not None:
            rel = "↑上方" if current_price > val else "↓下方"
            ma_parts.append(f"{label}={val:.2f}({rel})")

    valid_mas = [v for v in [ma5, ma10, ma20, ma60] if v is not None]
    if len(valid_mas) >= 3:
        if all(valid_mas[i] > valid_mas[i + 1] for i in range(len(valid_mas) - 1)):
            ma_order = "多头排列"
        elif all(valid_mas[i] < valid_mas[i + 1] for i in range(len(valid_mas) - 1)):
            ma_order = "空头排列"
        else:
            ma_order = "均线交叉/缠绕"
    else:
        ma_order = ""

    # MA5/MA20 金死叉（前一根 vs 最新）
    ma_cross = ""
    if n >= 21 and ma20 is not None:
        prev_ma5  = close.iloc[-6:-1].mean()
        prev_ma20 = close.iloc[-21:-1].mean()
        if prev_ma5 < prev_ma20 and ma5 >= ma20:
            ma_cross = "⚠️ MA5上穿MA20（金叉）"
        elif prev_ma5 > prev_ma20 and ma5 <= ma20:
            ma_cross = "⚠️ MA5下穿MA20（死叉）"

    # ── 2. MACD (12,26,9) ────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif   = ema12 - ema26
    dea   = dif.ewm(span=9, adjust=False).mean()
    hist  = (dif - dea) * 2

    dif_v     = dif.iloc[-1]
    dea_v     = dea.iloc[-1]
    hist_v    = hist.iloc[-1]
    hist_prev = hist.iloc[-2] if n >= 2 else hist_v
    dif_prev  = dif.iloc[-2]  if n >= 2 else dif_v
    dea_prev  = dea.iloc[-2]  if n >= 2 else dea_v

    macd_cross = ""
    if dif_prev < dea_prev and dif_v >= dea_v:
        macd_cross = "⚠️ MACD金叉（DIF上穿DEA）"
    elif dif_prev > dea_prev and dif_v <= dea_v:
        macd_cross = "⚠️ MACD死叉（DIF下穿DEA）"

    above_zero = "零轴上方" if dif_v > 0 else "零轴下方"
    if hist_v > 0:
        hist_desc = "红柱" + ("扩大" if hist_v > hist_prev else "缩小")
    else:
        hist_desc = "绿柱" + ("扩大" if hist_v < hist_prev else "缩小")

    # ── 3. RSI (14) ──────────────────────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    rsi14    = (100 - 100 / (1 + rs)).iloc[-1]

    if rsi14 >= 80:
        rsi_status = "强超买区（>80）"
    elif rsi14 >= 70:
        rsi_status = "超买区（70~80，注意回调）"
    elif rsi14 <= 20:
        rsi_status = "强超卖区（<20）"
    elif rsi14 <= 30:
        rsi_status = "超卖区（20~30，关注反弹）"
    elif rsi14 >= 50:
        rsi_status = "偏强（50~70）"
    else:
        rsi_status = "偏弱（30~50）"

    # ── 4. KDJ (9,3,3) ───────────────────────────────────────────
    low_min  = low.rolling(9, min_periods=1).min()
    high_max = high.rolling(9, min_periods=1).max()
    denom    = (high_max - low_min).replace(0, float("nan"))
    rsv      = ((close - low_min) / denom * 100).fillna(50)
    k_line   = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d_line   = k_line.ewm(alpha=1 / 3, adjust=False).mean()
    j_line   = 3 * k_line - 2 * d_line
    k_v, d_v, j_v = k_line.iloc[-1], d_line.iloc[-1], j_line.iloc[-1]

    if j_v > 90:
        kdj_status = "J值超买（>90）"
    elif j_v < 10:
        kdj_status = "J值超卖（<10）"
    elif k_v > d_v:
        kdj_status = "K在D上方（偏多）"
    else:
        kdj_status = "K在D下方（偏空）"

    # ── 5. 布林带 (20,2) ─────────────────────────────────────────
    if n >= 20:
        bb_mid   = close.rolling(20).mean().iloc[-1]
        bb_std   = close.rolling(20).std(ddof=1).iloc[-1]
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid * 100 if bb_mid > 0 else 0

        if current_price >= bb_upper * 0.99:
            bb_pos = "触及/突破上轨（强势或超买信号）"
        elif current_price <= bb_lower * 1.01:
            bb_pos = "触及/跌破下轨（弱势或超卖信号）"
        else:
            pct_in_band = (
                (current_price - bb_lower) / (bb_upper - bb_lower) * 100
                if (bb_upper - bb_lower) > 0 else 50
            )
            bb_pos = f"布林带内部{pct_in_band:.0f}%位置"

        bb_line = (
            f"上轨{bb_upper:.2f} / 中轨{bb_mid:.2f} / 下轨{bb_lower:.2f}，"
            f"{bb_pos}，带宽{bb_width:.1f}%"
        )
    else:
        bb_line = "（数据不足，无法计算布林带）"

    # ── 6. 成交量 ─────────────────────────────────────────────────
    vol5  = volume.tail(5).mean()
    vol20 = volume.tail(20).mean() if n >= 20 else volume.mean()
    vol_ratio = vol5 / vol20 if vol20 > 0 else 1.0

    if vol_ratio >= 2.0:
        vol_desc = f"近5日均量是20日均量的{vol_ratio:.1f}倍，异常放量"
    elif vol_ratio >= 1.5:
        vol_desc = f"近5日均量是20日均量的{vol_ratio:.1f}倍，显著放量"
    elif vol_ratio >= 1.2:
        vol_desc = f"近5日均量是20日均量的{vol_ratio:.1f}倍，温和放量"
    elif vol_ratio <= 0.5:
        vol_desc = f"近5日均量是20日均量的{vol_ratio:.1f}倍，深度缩量"
    elif vol_ratio <= 0.8:
        vol_desc = f"近5日均量是20日均量的{vol_ratio:.1f}倍，明显缩量"
    else:
        vol_desc = f"近5日均量与20日均量相当（比值{vol_ratio:.2f}），量能平稳"

    last_vol_ratio = volume.iloc[-1] / vol20 if vol20 > 0 else 1.0
    last_vol_tag   = (
        "明显放量" if last_vol_ratio >= 1.5
        else "缩量" if last_vol_ratio <= 0.7
        else "正常"
    )
    last_vol_desc = f"最新一根成交量为20日均量的{last_vol_ratio:.1f}倍（{last_vol_tag}）"

    # ── 7. 价格区间与涨跌幅 ───────────────────────────────────────
    period_high   = close.max()
    period_low    = close.min()
    price_range   = period_high - period_low
    position_pct  = (current_price - period_low) / price_range * 100 if price_range > 0 else 50.0
    recent20_high = high.tail(20).max() if n >= 20 else high.max()
    recent20_low  = low.tail(20).min()  if n >= 20 else low.min()

    chg_5d  = (current_price - close.iloc[-5])  / close.iloc[-5]  * 100 if n >= 5  else 0.0
    chg_20d = (current_price - close.iloc[-20]) / close.iloc[-20] * 100 if n >= 20 else 0.0
    chg_all = (current_price - close.iloc[0])   / close.iloc[0]   * 100 if n >= 2  else 0.0

    # ── 8. K线形态 ────────────────────────────────────────────────
    recent5   = df.tail(5)
    up_days   = int((recent5["close"] >= recent5["open"]).sum())
    down_days = 5 - up_days

    # 连续涨跌计数
    streak, streak_dir = 1, ("up" if df.iloc[-1]["close"] >= df.iloc[-1]["open"] else "down")
    for i in range(len(df) - 2, max(len(df) - 10, -1), -1):
        d = "up" if df.iloc[i]["close"] >= df.iloc[i]["open"] else "down"
        if d == streak_dir:
            streak += 1
        else:
            break
    streak_desc = (
        f"连续{'上涨' if streak_dir == 'up' else '下跌'}{streak}根K线"
        if streak >= 2 else "近期无连续涨跌"
    )

    # 最新K线：实体 + 上下影线
    last      = df.iloc[-1]
    body      = abs(last["close"] - last["open"])
    body_pct  = body / last["open"] * 100 if last["open"] > 0 else 0
    upper_shd = last["high"] - max(last["close"], last["open"])
    lower_shd = min(last["close"], last["open"]) - last["low"]
    candle_t  = "阳线" if last["close"] >= last["open"] else "阴线"

    if body_pct >= 3:
        candle_desc = f"大{candle_t}（实体{body_pct:.1f}%）"
    elif body_pct < 0.3:
        candle_desc = f"十字星/小实体（{body_pct:.1f}%），多空分歧"
    else:
        candle_desc = f"{candle_t}（实体{body_pct:.1f}%）"

    if upper_shd > body * 1.5 and body > 0:
        candle_desc += "，上影线长（上方压力大）"
    if lower_shd > body * 1.5 and body > 0:
        candle_desc += "，下影线长（下方支撑强）"

    # ── 拼装 ─────────────────────────────────────────────────────
    summary = (
        f"【{stock_name}】技术面综合摘要（近{n}个交易日日K）：\n"
        f"\n▌ 价格与区间\n"
        f"- 当前价：{current_price:.2f}元\n"
        f"- 近{n}日区间：{period_low:.2f} ~ {period_high:.2f}，当前处于{position_pct:.1f}%分位\n"
        f"- 近20日关键位：压力{recent20_high:.2f} / 支撑{recent20_low:.2f}\n"
        f"- 涨跌幅：近5日{chg_5d:+.2f}%，近20日{chg_20d:+.2f}%，期间累计{chg_all:+.2f}%\n"
        f"\n▌ 均线系统\n"
        f"- {' / '.join(ma_parts)}\n"
        + (f"- 排列形态：{ma_order}\n" if ma_order else "")
        + (f"- {ma_cross}\n" if ma_cross else "")
        + f"\n▌ MACD (12,26,9)\n"
        f"- DIF={dif_v:+.3f}，DEA={dea_v:+.3f}，柱={hist_v:+.3f}（{above_zero}，{hist_desc}）\n"
        + (f"- {macd_cross}\n" if macd_cross else "")
        + f"\n▌ RSI & KDJ\n"
        f"- RSI(14)={rsi14:.1f}，{rsi_status}\n"
        f"- KDJ：K={k_v:.1f}，D={d_v:.1f}，J={j_v:.1f}，{kdj_status}\n"
        f"\n▌ 布林带 (20,2)\n"
        f"- {bb_line}\n"
        f"\n▌ 成交量\n"
        f"- {vol_desc}\n"
        f"- {last_vol_desc}\n"
        f"\n▌ K线形态（近5日）\n"
        f"- {up_days}阳{down_days}阴，{streak_desc}\n"
        f"- 最新K线：{candle_desc}\n"
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
    current_price: float = _safe_float(basic["data"].get("f43", 0), 0.0)

    raw_chg_amt = basic["data"].get("f169", 0)
    raw_chg_pct = basic["data"].get("f170", 0)
    chg_amount: float = _safe_float(raw_chg_amt, 0.0)
    chg_pct: float = _safe_float(raw_chg_pct, 0.0)

    # 获取日K（近60个交易日，用于图表展示）
    kline_raw = await get_gg(sec_id, "single-stock-kline-101")
    if isinstance(kline_raw, str):
        return None, None, None, None, kline_raw, 0.0, 0.0, 0.0

    df = _parse_kline_df(kline_raw)
    if df is None or df.empty:
        return None, None, None, None, "无有效K线数据", 0.0, 0.0, 0.0

    # `single-stock-kline-101` 已按近半年窗口请求，保留整段数据给图表与技术面摘要
    df = df.reset_index(drop=True)

    tech_summary = _build_tech_summary(df, stock_name, current_price)
    logger.info(f"[SayuStock] 技术面摘要生成完成: {stock_name}({stock_code}) [{market_name}]")
    return stock_name, stock_code, market_name, df, tech_summary, current_price, chg_amount, chg_pct
