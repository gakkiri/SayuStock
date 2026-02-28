from gsuid_core.sv import SV
from gsuid_core.bot import Bot
from gsuid_core.logger import logger
from gsuid_core.models import Event

from .get_data import get_stock_tech_data
from .llm_analyze import kimi_analyze
from .draw_result import draw_analysis_img

sv_stock_analysis = SV("大盘分析")


@sv_stock_analysis.on_prefix(("大盘分析",))
async def send_stock_analysis(bot: Bot, ev: Event):
    code = ev.text.strip()
    if not code:
        await bot.send("请输入股票名称或代码，例如：大盘分析 茅台")
        return

    logger.info(f"[SayuStock] 开始执行[大盘分析] code={code}")
    await bot.send(f"正在分析【{code}】，请稍候...")

    try:
        logger.info("[SayuStock][大盘分析] 第1步: 获取技术面数据")
        stock_name, stock_code, market_name, df, tech_summary, cur_price, chg_amt, chg_pct = await get_stock_tech_data(code)
        if df is None:
            logger.warning(f"[SayuStock][大盘分析] 获取数据失败: {tech_summary}")
            await bot.send(tech_summary or "获取数据失败")
            return
        logger.info(f"[SayuStock][大盘分析] 技术面数据OK: {stock_name}({stock_code}) [{market_name}], K线{len(df)}条")

        logger.info("[SayuStock][大盘分析] 第2步: 调用 Kimi 分析")
        verdict, reason, risk, news_summary, sources = await kimi_analyze(stock_name, tech_summary)
        logger.info(f"[SayuStock][大盘分析] Kimi结果: verdict={verdict}, sources={sources}")

        logger.info("[SayuStock][大盘分析] 第3步: 渲染图片")
        im = await draw_analysis_img(
            stock_name, stock_code, market_name, df,
            cur_price, chg_amt, chg_pct,
            verdict, reason, risk, news_summary, sources,
        )
        logger.info("[SayuStock][大盘分析] 渲染完成，发送图片")

        await bot.send(im, at_sender=True)

    except Exception as e:
        logger.exception(f"[SayuStock][大盘分析] 出现异常: {e}")
        await bot.send(f"分析出错: {e}")

