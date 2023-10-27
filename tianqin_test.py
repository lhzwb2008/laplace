from datetime import date
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask
# 在创建 api 实例时传入 TqBacktest 就会进入回测模式,设置web_gui=True开启图形化界面
api = TqApi(backtest=TqBacktest(start_dt=date(2018, 5, 2), end_dt=date(2018, 6, 2)),web_gui=True, auth=TqAuth("zark2023", "Hello2023"))
# 获得 m1901 5分钟K线的引用
klines = api.get_kline_serial("DCE.m1901", 5 * 60, data_length=15)
# 创建 m1901 的目标持仓 task，该 task 负责调整 m1901 的仓位到指定的目标仓位
target_pos = TargetPosTask(api, "DCE.m1901")
while True:
    api.wait_update()
    if api.is_changing(klines):
        ma = sum(klines.close.iloc[-15:]) / 15
        print("最新价", klines.close.iloc[-1], "MA", ma)
        if klines.close.iloc[-1] > ma:
            print("最新价大于MA: 目标多头5手")
            # 设置目标持仓为多头5手
            target_pos.set_target_volume(5)
        elif klines.close.iloc[-1] < ma:
            print("最新价小于MA: 目标空仓")
            # 设置目标持仓为空仓
            target_pos.set_target_volume(0)