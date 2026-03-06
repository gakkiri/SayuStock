from gsuid_core.sv import SV
from gsuid_core.aps import scheduler
from gsuid_core.bot import Bot
from gsuid_core.logger import logger
from gsuid_core.models import Event

from .draw_info import draw_info_img
from .draw_future import draw_future_img
from .draw_my_info import draw_my_stock_img, draw_user_stock_img
from .draw_fund_info import draw_fund_info

sv_stock_info = SV("大盘概览")
sv_my_stock = SV("我的自选")
sv_fund_info = SV("基金持仓信息")


@sv_fund_info.on_command(("基金持仓", "持仓分布"), block=True)
async def send_fund_info(bot: Bot, ev: Event):
    logger.info("[SayuStock] 开始执行[基金持仓信息]")
    im = await draw_fund_info(ev.text.strip())
    await bot.send(im)


@sv_stock_info.on_fullmatch(("大盘概览", "大盘概况"))
async def send_stock_info(bot: Bot, ev: Event):
    logger.info("[SayuStock] 开始执行[大盘概览]")
    im = await draw_info_img()
    await bot.send(im)


@sv_my_stock.on_fullmatch(("我的自选", "我的持仓", "我的股票"))
async def send_my_stock(bot: Bot, ev: Event):
    logger.info("[SayuStock] 开始执行[我的自选]")
    await bot.send(await draw_my_stock_img(ev))


@sv_my_stock.on_regex(r"^(\d{5,20})的自选$", block=True)
async def send_other_user_stock(bot: Bot, ev: Event):
    target_qq = ev.regex_group[0] if ev.regex_group else ""
    if not target_qq:
        return await bot.send("请输入正确的QQ号，例如：12345678的自选")

    logger.info(f"[SayuStock] 开始执行[{target_qq}的自选]")
    await bot.send(
        await draw_user_stock_img(
            target_qq,
            ev.bot_id,
            f"QQ号 {target_qq} 还未添加自选呢~请输入 添加自选 查看帮助!",
        )
    )


@sv_my_stock.on_fullmatch(("全天候", "全天候板块"))
async def send_future_stock(bot: Bot, ev: Event):
    logger.info("[SayuStock] 开始执行[全天候板块]")
    await bot.send(await draw_future_img())


# 每日晚上十一点保存当天数据
@scheduler.scheduled_job("cron", hour=23, minute=0)
async def save_data_sayustock():
    await draw_info_img(is_save=True)
