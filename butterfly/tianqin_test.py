from tqsdk import TqApi, TqAuth,TqSim,TargetPosTask,TqAccount
import time
# api = TqApi(TqAccount("H徽商期货", "952522", "Hello2023"), auth=TqAuth("卡卡罗特2023", "Hello2023"))
sim = TqSim(init_balance=20000)
api = TqApi(sim,auth=TqAuth("卡卡罗特2023", "Hello2023"))
account = api.get_account()
while True:
    print(account.balance)
    print(account.float_profit)
    time.sleep(60)

