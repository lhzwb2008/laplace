# from tqsdk import TqAccount, TqApi, TqAuth,TqKq
# # api = TqApi(TqAccount("H徽商期货", "952522", "Hello2023"), auth=TqAuth("卡卡罗特2023", "Hello2023"))
# api = TqApi(TqKq(), auth=TqAuth("卡卡罗特2023", "Hello2023"))
# print(api.get_account())
# api.close()

from tqsdk import TqApi, TqAuth,TqSim,TargetPosTask
import datetime
import pandas as pd

def get_price(direction):
    # 在 BUY 时使用买一价加一档价格，SELL 时使用卖一价减一档价格
    if direction == "BUY":
        price = quote.bid_price1 + quote.price_tick
    else:
        price = quote.ask_price1 - quote.price_tick
    # 如果 price 价格是 nan，使用最新价报单
    if price != price:
        price = quote.last_price
    return price
sim = TqSim(init_balance=100000)
api = TqApi(sim,auth=TqAuth("卡卡罗特2023", "Hello2023"))
print(sim.get_margin("SHFE.ss2312"))

# 获得 i2209 tick序列的引用
ticks = api.get_tick_serial("SHFE.ss2312")
lock = 0

while True:
    api.wait_update()
    # 判断整个tick序列是否有变化
    if api.is_changing(ticks) and lock==0:
        lock=1
        volume = 10
        tick = ticks.iloc[-1].to_dict()
        account = api.get_account()
        position = api.get_position("SHFE.ss2312")
        quote = api.get_quote("SHFE.ss2312")
        target_pos = TargetPosTask(api, "SHFE.ss2312", price=get_price)
        # order = api.insert_order(symbol="SHFE.ss2312", direction="BUY", offset="OPEN", volume=volume,limit_price=quote.ask_price1)
        target_pos.set_target_volume(volume)
        while True:
            if position.pos_long == volume:
                break
            api.wait_update()
            print(account)  
            print(position) 
        lock=0

