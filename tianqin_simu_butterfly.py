import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,precision_score
from tqsdk import TqApi, TqAuth,TqSim,TargetPosTask,TqAccount
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')


# 加载数据
FILENAME = "future_taobao_ssMain_tick"
data = pd.read_csv(FILENAME+"_with_opportunities.csv")

data_clean = data.dropna().copy()  # 创建一个副本以避免警告

features = [
    'last_price','highest','lowest', 'volume', 'open_interest', 'volume_delta', 'open_interest_delta',
    'bid_price1', 'ask_price1', 'bid_volume1', 'ask_volume1'
]
# 目标列处理
data_clean['successful_trade'] = data_clean.apply(
    lambda row: 1 if row['buy_opportunity'] and row['sell_opportunity'] else 0, axis=1
)

X = data_clean[['datetime'] + features]
y = data_clean['successful_trade']

# 数据切分
split_start = int(len(data_clean) * 0)
split_point = int(len(data_clean) * 1)
X_train = X.iloc[split_start:split_point]
y_train = y.iloc[split_start:split_point]
X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]

# 模型训练
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# 注意：随机森林模型不能直接处理datetime类型的特征，所以在训练前需要去除或转换该特征
X_train_features = X_train.drop(columns=['datetime'])
X_test_features = X_test.drop(columns=['datetime'])
rf_model.fit(X_train_features, y_train)



class DualWriter:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):  # flush方法是为了兼容sys.stdout
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        
def parse_time_range(time_range_str):
    """解析时间范围字符串并返回时间对象的开始和结束时间"""
    start_str, end_str = time_range_str.split('-')
    start_time = datetime.strptime(start_str, "%H:%M").time()
    end_time = datetime.strptime(end_str, "%H:%M").time()
    return start_time, end_time

def is_time_in_ranges(time_to_check, time_ranges):
    """判断给定时间是否在时间范围数组内"""
    for time_range in time_ranges:
        start_time, end_time = parse_time_range(time_range)
        if start_time <= time_to_check <= end_time:
            return True
    return False

# 定义时间范围数组
notrade_time = ["09:00-09:10","11:20-11:30","13:30-13:40","14:50-15:00","21:00-21:10","0:00-1:00"]

logger = DualWriter('tianqin_simu.log')
# 保存原始的stdout
original_stdout = sys.stdout
# 重定向stdout到我们的DualWriter
sys.stdout = logger


future_code = "SHFE.ss2405"
# sim = TqSim(init_balance=10000)
# sim.set_commission(future_code, 2)
# api = TqApi(sim,auth=TqAuth("卡卡罗特2023", "Hello2023"))
api = TqApi(TqAccount("H徽商期货", "952522", "Hello2023"), auth=TqAuth("卡卡罗特2023", "Hello2023"))
ticks = api.get_tick_serial(future_code)
quote = api.get_quote(future_code)

tick = pd.DataFrame()

last_datetime = None  # 用于存储上次循环中最后一个tick的时间戳
lock = 1 

last_volume = 0
last_open_interest = 0
trade_threshold = 0.6
trade_hand = 1
guess_tick=100

while True:
    api.wait_update()
    # 判断整个tick序列是否有变化
    if api.is_changing(ticks): 
        tick = ticks.iloc[-1]
        if last_datetime is not None:
            last_datetime = tick['datetime']
            tick['volume_delta'] = tick['volume']-last_volume
            tick['open_interest_delta'] = tick['open_interest'] = last_open_interest
            last_volume = tick['volume']
            last_open_interest = tick['open_interest']
        else:
            last_datetime = tick['datetime']
            last_volume = tick['volume']
            last_open_interest = tick['open_interest']
            continue

        if tick['volume_delta']==0:
            continue
        #不交易时段
        if is_time_in_ranges(datetime.fromtimestamp(last_datetime/1_000_000_000).time(),notrade_time):
            continue
        
        account = api.get_account()
        position = api.get_position(future_code)
        if position.pos_long > trade_hand or position.pos_short > trade_hand:
            print("exceed trade_hand")
            break
        
        tick_df = tick.to_frame().T
        tick_features = tick_df[features]
        probability = rf_model.predict_proba(tick_features)[:, 1]
        if lock > 0 and probability is not None and probability>trade_threshold:
            lock = lock - 1
            if position.pos_long == 0 and position.pos_short == 0:
                print("start open")
                print(datetime.now())
                #双开
                order_buy = api.insert_order(symbol=future_code, direction="BUY", offset="OPEN", limit_price=tick['bid_price1'], volume=trade_hand)
                order_sell = api.insert_order(symbol=future_code, direction="SELL", offset="OPEN", limit_price=tick['ask_price1'], volume=trade_hand)
                start_time = datetime.now()
                tick_count = 0
                while True:
                    api.wait_update()
                    end_time = datetime.now()
                    count_ticks = api.get_tick_serial(future_code)
                    if api.is_changing(count_ticks): 
                        count_tick = count_ticks.iloc[-1]
                        count_tick['volume_delta'] = count_tick['volume']-last_volume
                        last_volume = count_tick['volume']
                        if count_tick['volume_delta'] == 0:
                            continue
                        tick_count = tick_count + 1 
                    if position.pos_long == trade_hand and position.pos_short == trade_hand:
                        print(account.balance)
                        lock = lock + 1
                        break
                    # if (end_time - start_time).total_seconds() > 300:
                    if tick_count >= guess_tick:
                        print("after guess_tick faild")
                        print((end_time - start_time).total_seconds())
                        if position.pos_long == trade_hand and position.pos_short == 0: 
                            api.cancel_order(order_sell)
                            order = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=count_tick['bid_price1'], volume=trade_hand) 
                            lock = lock + 1
                            print(account.balance)
                            break
                        elif position.pos_long == 0 and position.pos_short == trade_hand:
                            api.cancel_order(order_buy)
                            order = api.insert_order(symbol=future_code, direction="BUY", offset="CLOSETODAY", limit_price=count_tick['ask_price1'], volume=trade_hand) 
                            lock = lock + 1
                            print(account.balance)
                            break
                        else:
                            api.cancel_order(order_buy)
                            api.cancel_order(order_sell)
                            lock = lock + 1
                            break
            elif position.pos_long == trade_hand and position.pos_short == trade_hand:
                #双平
                print("start close")
                print(datetime.now())
                order_buy = api.insert_order(symbol=future_code, direction="BUY", offset="CLOSETODAY", limit_price=tick['bid_price1'], volume=trade_hand)
                order_sell = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=tick['ask_price1'], volume=trade_hand)
                start_time = datetime.now()
                tick_count = 0
                while True:
                    api.wait_update()
                    end_time = datetime.now()
                    count_ticks = api.get_tick_serial(future_code)
                    if api.is_changing(count_ticks): 
                        count_tick = count_ticks.iloc[-1]
                        count_tick['volume_delta'] = count_tick['volume']-last_volume
                        last_volume = count_tick['volume']
                        if count_tick['volume_delta'] == 0:
                            continue
                        tick_count = tick_count + 1 
                    if position.pos_long == 0 and position.pos_short == 0:
                        lock = lock + 1
                        print(account.balance)
                        break
                    # if (end_time - start_time).total_seconds() > 300:
                    if tick_count >= guess_tick:
                        print("after guess_tick faild")
                        print((end_time - start_time).total_seconds())
                        if position.pos_long == trade_hand and position.pos_short == 0: 
                            api.cancel_order(order_sell)
                            order = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=count_tick['bid_price1'], volume=trade_hand) 
                            lock = lock + 1
                            print(account.balance)
                            break
                        elif position.pos_long == 0 and position.pos_short == trade_hand:
                            api.cancel_order(order_buy)
                            order = api.insert_order(symbol=future_code, direction="BUY", offset="CLOSETODAY", limit_price=count_tick['ask_price1'], volume=trade_hand) 
                            lock = lock + 1
                            print(account.balance)
                            break
                        else:
                            api.cancel_order(order_buy)
                            api.cancel_order(order_sell)
                            lock = lock + 1
                            break
                    
        