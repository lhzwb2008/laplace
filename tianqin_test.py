from tqsdk import TqAccount, TqApi, TqAuth,TqKq
api = TqApi(TqAccount("H徽商期货", "952522", "Hello2023"), auth=TqAuth("卡卡罗特2023", "Hello2023"))
klines = api.get_kline_serial("SHFE.ss2401", 1)
while True:
    api.wait_update()
    order = api.insert_order(symbol="SHFE.ss2401", direction="SELL", offset="CLOSETODAY", limit_price=14045, volume=1)

