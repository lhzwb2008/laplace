from tqsdk import TqAccount, TqApi, TqAuth,TqKq
from datetime import date 
# api = TqApi(TqAccount("H徽商期货", "952522", "Hello2023"), auth=TqAuth("卡卡罗特2023", "Hello2023"))
api = TqApi(TqKq(), auth=TqAuth("卡卡罗特2023", "Hello2023"))
conts = api.query_his_cont_quotes(symbol=['KQ.m@SHFE.ss'], n=20)
print(conts)
api.close()
    


