"""渲染分析结果：生成完整 HTML 后用 Playwright 一次截图"""
import re
import html as html_escape_lib
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from playwright.async_api import async_playwright

try:
    from gsuid_core.utils.image.convert import convert_img
except ImportError:
    import io
    from PIL import Image
    async def convert_img(data):  # type: ignore
        if isinstance(data, bytes):
            return data
        buf = io.BytesIO()
        data.save(buf, "PNG")
        return buf.getvalue()

CACHE_PATH = Path(__file__).parent / "cache"
CACHE_PATH.mkdir(exist_ok=True)

UP_COLOR = "#f85149"
DOWN_COLOR = "#3fb950"


def _build_fig(df: pd.DataFrame) -> go.Figure:
    colors = [UP_COLOR if r["close"] >= r["open"] else DOWN_COLOR for _, r in df.iterrows()]
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing=dict(line=dict(color=UP_COLOR, width=1), fillcolor=UP_COLOR),
        decreasing=dict(line=dict(color=DOWN_COLOR, width=1), fillcolor=DOWN_COLOR),
        name="K线", showlegend=False,
    ), row=1, col=1)

    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df["date"], y=ma5,
        line=dict(color="#d29922", width=1.2), name="MA5",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["date"], y=ma20,
        line=dict(color="#a78bfa", width=1.2), name="MA20",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"],
        marker_color=colors, name="成交量", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(
            orientation="h", x=0.5, y=1.04, xanchor="center",
            font=dict(size=12, color="#c9d1d9"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=560,
    )
    axis_style = dict(gridcolor="#21262d", zerolinecolor="#21262d",
                      tickfont=dict(color="#8b949e", size=10))
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _md_to_html(text: str) -> str:
    """把 Kimi 返回的简单 Markdown 转成 HTML 片段（不依赖外部库）。"""
    # 先转义 HTML 特殊字符，再处理 Markdown 语法（* _ 不是 HTML 特殊字符，不受影响）
    text = html_escape_lib.escape(text)
    # **bold**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # *italic* 或 _italic_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    # `code`
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # 列表项：行首 - 或 · 转为带 • 的行
    text = re.sub(r'(?m)^[-·]\s+', '<span class="li-dot">•</span> ', text)
    # 换行
    text = text.replace('\n', '<br>')
    return text


def _verdict_class(verdict: str) -> str:
    if any(w in verdict for w in ["多", "看涨", "买入", "偏多"]):
        return "bullish"
    if any(w in verdict for w in ["空", "看跌", "卖出", "偏空"]):
        return "bearish"
    return "neutral"


def _build_html(
    stock_name: str,
    stock_code: str,
    market_name: str,
    fig: go.Figure,
    cur_price: float,
    chg_amount: float,
    chg_pct: float,
    verdict: str,
    reason: str,
    risk: str,
    news_summary: str,
    sources: List[str],
) -> str:
    chart_fragment = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displayModeBar": False},
        div_id="kline-div",
    )
    vc = _verdict_class(verdict)
    market_html = (
        f'<div class="market-tag">{html_escape_lib.escape(market_name)}</div>'
        if market_name else ""
    )
    # 当前价 + 涨跌幅
    chg_sign = "+" if chg_amount >= 0 else ""
    price_class = "price-up" if chg_amount >= 0 else "price-down"
    price_html = (
        f'<div class="price-info">'
        f'<span class="cur-price">¥{cur_price:.2f}</span>'
        f'<span class="{price_class}">{chg_sign}{chg_amount:.2f}　{chg_sign}{chg_pct:.2f}%</span>'
        f'</div>'
    ) if cur_price else ""
    news_summary_html = (
        f'<div class="news-summary">{_md_to_html(news_summary)}</div>'
        if news_summary else ""
    )
    sources_html = "".join(
        f'<div class="source-item"><span class="ref-num">[{i+1}]</span>{html_escape_lib.escape(s)}</div>'
        for i, s in enumerate(sources[:6])
    ) or '<div class="source-item" style="color:#484f58">暂无信源</div>'

    reason_html = _md_to_html(reason)
    risk_html = _md_to_html(risk)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    width: 900px;
  }}
  .header {{
    display: flex;
    align-items: center;
    padding: 14px 20px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
  }}
  .header-title {{
    font-size: 18px;
    font-weight: 700;
    color: #e6edf3;
  }}
  .stock-code {{
    margin-left: 8px;
    font-size: 13px;
    color: #8b949e;
    font-family: monospace;
  }}
  .market-tag {{
    margin-left: 10px;
    font-size: 11px;
    font-weight: 600;
    color: #388bfd;
    background: rgba(56,139,253,0.12);
    border: 1px solid rgba(56,139,253,0.3);
    border-radius: 4px;
    padding: 2px 7px;
    letter-spacing: 0.5px;
  }}
  .price-info {{
    margin-left: 16px;
    display: flex;
    align-items: baseline;
    gap: 8px;
  }}
  .cur-price {{
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
  }}
  .price-up {{ font-size: 13px; color: #f85149; }}
  .price-down {{ font-size: 13px; color: #3fb950; }}
  .header-sub {{
    margin-left: auto;
    font-size: 11px;
    color: #484f58;
  }}

  /* 第一段：K线 */
  .chart-area {{
    background: #0d1117;
    padding: 8px 4px 0 4px;
  }}

  /* 第二段：分析结论 */
  .analysis {{
    background: #161b22;
    border-top: 1px solid #30363d;
    padding: 0;
  }}
  .verdict {{
    padding: 14px 20px;
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1px;
    border-bottom: 1px solid #21262d;
  }}
  .verdict.bullish {{ background: rgba(248,81,73,0.1); color: #f85149; border-left: 4px solid #f85149; }}
  .verdict.bearish {{ background: rgba(63,185,80,0.1); color: #3fb950; border-left: 4px solid #3fb950; }}
  .verdict.neutral  {{ background: rgba(210,153,34,0.1); color: #d29922; border-left: 4px solid #d29922; }}
  .analysis-body {{
    display: flex;
    gap: 0;
  }}
  .analysis-col {{
    flex: 1;
    padding: 16px 20px;
  }}
  .analysis-col + .analysis-col {{
    border-left: 1px solid #21262d;
  }}
  .col-title {{
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
  }}
  .col-body {{
    font-size: 14px;
    line-height: 1.75;
    color: #c9d1d9;
  }}
  .risk-body {{
    font-size: 13px;
    line-height: 1.75;
    color: #f0883e;
  }}

  /* 第三段：引用信源 */
  .sources {{
    background: #0d1117;
    border-top: 1px solid #30363d;
    padding: 12px 20px 14px;
  }}
  .sources-title {{
    font-size: 11px;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
  }}
  .news-summary {{
    font-size: 13px;
    color: #c9d1d9;
    line-height: 1.7;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
  }}
  .source-item {{
    font-size: 12px;
    color: #8b949e;
    padding: 2px 0;
    line-height: 1.6;
  }}
  .ref-num {{
    color: #388bfd;
    margin-right: 6px;
    font-size: 11px;
  }}
  .li-dot {{ color: #388bfd; margin-right: 4px; }}
  code {{ background: #21262d; padding: 1px 5px; border-radius: 3px; font-size: 12px; }}

  .footer {{
    font-size: 11px;
    color: #484f58;
    text-align: right;
    padding: 8px 20px;
    border-top: 1px solid #21262d;
    background: #0d1117;
  }}
</style>
</head>
<body>
  <div class="header">
    <div class="header-title">【{html_escape_lib.escape(stock_name)}】综合分析</div>
    <div class="stock-code">{html_escape_lib.escape(stock_code)}</div>
    {market_html}
    {price_html}
    <div class="header-sub">数据：东方财富 &nbsp;|&nbsp; 日K · 近60日</div>
  </div>

  <div class="chart-area">
    {chart_fragment}
  </div>

  <div class="analysis">
    <div class="verdict {vc}">{html_escape_lib.escape(verdict)}</div>
    <div class="analysis-body">
      <div class="analysis-col">
        <div class="col-title">核心理由</div>
        <div class="col-body">{reason_html}</div>
      </div>
      <div class="analysis-col">
        <div class="col-title">风险提示</div>
        <div class="risk-body">{risk_html}</div>
      </div>
    </div>
  </div>

  <div class="sources">
    <div class="sources-title">参考信源</div>
    {news_summary_html}
    {sources_html}
  </div>

  <div class="footer">以上分析由 AI 自动生成，仅供参考，不构成投资建议</div>
</body>
</html>"""


async def _render(html_path: Path) -> bytes:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 900, "height": 900},
            device_scale_factor=2,
        )
        page = await context.new_page()
        await page.goto(html_path.absolute().as_uri())
        await page.wait_for_selector(".plot-container", timeout=15000)
        await page.wait_for_timeout(800)
        png = await page.screenshot(type="png", full_page=True)
        await browser.close()
    return png


async def draw_analysis_img(
    stock_name: str,
    stock_code: str,
    market_name: str,
    df: pd.DataFrame,
    cur_price: float,
    chg_amount: float,
    chg_pct: float,
    verdict: str,
    reason: str,
    risk: str,
    news_summary: str,
    sources: List[str],
) -> bytes:
    fig = _build_fig(df)
    html_content = _build_html(
        stock_name, stock_code, market_name, fig,
        cur_price, chg_amount, chg_pct,
        verdict, reason, risk, news_summary, sources,
    )
    html_path = CACHE_PATH / f"{stock_name}_analysis.html"
    html_path.write_text(html_content, encoding="utf-8")
    png = await _render(html_path)
    return await convert_img(png)
