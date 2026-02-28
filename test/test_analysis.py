"""
独立测试：大盘分析核心逻辑（不依赖 gsuid_core 框架）
运行：cd SayuStock插件根目录 && python3 test/test_analysis.py 茅台
"""
import sys
import asyncio
from pathlib import Path

import aiohttp
import pandas as pd

# 把 SayuStock 包目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from SayuStock.stock_analysis.get_data import _parse_kline_df, _build_tech_summary
from SayuStock.stock_analysis.llm_analyze import kimi_analyze
from SayuStock.stock_analysis.draw_result import draw_analysis_img

# 东财搜索：名称/代码 → secid
SEARCH_URL = "https://searchapi.eastmoney.com/api/suggest/get"
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
STOCK_URL = "https://push2.eastmoney.com/api/qt/stock/get"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


async def search_secid(keyword: str) -> tuple[str, str] | None:
    """搜索股票，返回 (secid, name)，如 ('1.600519', '贵州茅台')"""
    params = {
        "input": keyword,
        "type": "14",
        "token": "D43BF722C8E33BDC906FB84D85E326E8",
        "count": "5",
    }
    async with aiohttp.ClientSession(headers=HEADERS) as sess:
        async with sess.get(SEARCH_URL, params=params) as resp:
            data = await resp.json(content_type=None)
    items = data.get("QuotationCodeTable", {}).get("Data", [])
    if not items:
        return None
    item = items[0]
    # secid 格式：市场.代码，如 "1.600519"
    secid = f"{item['MktNum']}.{item['Code']}"
    name = item["Name"]
    return secid, name


async def fetch_kline(secid: str) -> dict:
    """拉日K线，近60天"""
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=100)
    params = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    async with aiohttp.ClientSession(headers=HEADERS) as sess:
        async with sess.get(KLINE_URL, params=params) as resp:
            return await resp.json(content_type=None)


async def fetch_basic(secid: str) -> dict:
    """拉基础行情（当前价等）"""
    fields = "f43,f58,f170,f44,f45"
    params = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields": fields,
        "invt": "2",
        "fltt": "2",
    }
    async with aiohttp.ClientSession(headers=HEADERS) as sess:
        async with sess.get(STOCK_URL, params=params) as resp:
            return await resp.json(content_type=None)


async def main(keyword: str):
    print(f"[1/4] 搜索股票: {keyword}")
    result = await search_secid(keyword)
    if result is None:
        print("未找到股票，请检查名称或代码")
        return
    secid, name = result
    print(f"      → {name}  secid={secid}")

    print("[2/4] 拉取 K线 + 基础行情")
    kline_raw, basic_raw = await asyncio.gather(fetch_kline(secid), fetch_basic(secid))

    # 解析价格
    basic_data = basic_raw.get("data") or {}
    current_price = float(basic_data.get("f43", 0))
    stock_name = basic_data.get("f58", name).split("(")[0].strip() or name
    print(f"      → 当前价: {current_price}")

    # 解析 K 线（复用 get_data 的纯函数）
    df = _parse_kline_df(kline_raw)
    if df is None or df.empty:
        print("K线数据为空，退出")
        return
    df = df.tail(60).reset_index(drop=True)
    print(f"      → K线条数: {len(df)}")

    tech_summary = _build_tech_summary(df, stock_name, current_price)
    print(f"\n--- 技术面摘要 ---\n{tech_summary}\n")

    print("[3/4] 调用 Kimi 分析（含网络搜索，约10-30秒）")
    verdict, reason, risk, sources = await kimi_analyze(stock_name, tech_summary)
    print(f"      判断: {verdict}")
    print(f"      理由: {reason}")
    print(f"      风险: {risk}")
    print(f"      信源: {sources}")

    print("[4/4] 渲染图片")
    img_bytes = await draw_analysis_img(stock_name, df, verdict, reason, risk, sources)
    out_path = Path(__file__).parent / f"output_{stock_name}.png"
    out_path.write_bytes(img_bytes)
    print(f"      → 图片已保存: {out_path}")


if __name__ == "__main__":
    keyword = sys.argv[1] if len(sys.argv) > 1 else "茅台"
    asyncio.run(main(keyword))
