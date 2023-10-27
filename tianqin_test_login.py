from tqsdk import TqAccount, TqApi, TqAuth

api = TqApi(TqAccount("H徽商期货", "952522", "Hello2023"), auth=TqAuth("卡卡罗特2023", "Hello2023"))
print(api.get_account())