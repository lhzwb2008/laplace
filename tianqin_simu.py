import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense,GRU,Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from keras.utils import Sequence
from tensorflow.keras.models import Sequential
from keras.models import load_model
from datetime import datetime, time
from tqsdk import TqApi, TqAuth,TqSim,TargetPosTask,TqAccount
import warnings
warnings.filterwarnings('ignore')


data_clean = pd.read_csv("future_taobao_ss2403_tick.csv")


# 1. 数据加载和预处理
data_clean['last_price'] = pd.to_numeric(data_clean['last_price'], errors='coerce')

pre_features = ['last_price','bid_price1','ask_price1','bid_price2','ask_price2','bid_price3','ask_price3','bid_price4','ask_price4','bid_price5','ask_price5','bid_volume1','bid_volume2','bid_volume3','bid_volume4','bid_volume5','ask_volume1','ask_volume2','ask_volume3','ask_volume4','ask_volume5']
for feature in pre_features:
    data_clean[feature + '_diff'] =  data_clean[feature].diff()
data_clean['last_price_bid_diff'] =  data_clean['last_price'] - data_clean['bid_price1']  
data_clean['last_price_ask_diff'] =  data_clean['last_price'] - data_clean['ask_price1']  
data_clean['last_price_highest_diff'] =  data_clean['last_price'] - data_clean['highest']  
data_clean['last_price_lowest_diff'] =  data_clean['last_price'] - data_clean['lowest']  
data_clean['datetime'] = pd.to_datetime(data_clean['datetime'])

# Initialize features list with pre_features
features = list(pre_features)

# Add difference features for each pre_feature
diff_features = [feature + '_diff' for feature in pre_features]
features.extend(diff_features)

# Add specific price difference features
additional_features = [
    'last_price_bid_diff', 'last_price_ask_diff', 'last_price_highest_diff', 'last_price_lowest_diff'
]
features.extend(additional_features)

# Define label
data_clean.dropna(subset=['bid_price1'], inplace=True)
data_clean['label'] = (data_clean['bid_price1'].shift(-30) > data_clean['bid_price1']).astype(int)

# 3. 分割数据
# Now you can filter the data between two dates
train_data = data_clean[(data_clean['datetime'] >= '2023-12-01 09:00:00') & 
                        (data_clean['datetime'] < '2024-01-15 09:00:00')]

test_data = data_clean[(data_clean['datetime'] >= '2024-01-15 09:00:00') & 
                        (data_clean['datetime'] < '2024-01-22 09:00:00')]


# 初始化归一化器
scaler = MinMaxScaler(feature_range=(0, 1))

train_data[features] = scaler.fit_transform(train_data[features])

# 将 DataFrame 转换为 NumPy 数组
X_train = np.array(train_data[features])
y_train = np.array(train_data['label'])

# 删除 NaN 值
mask = ~np.isnan(X_train).any(axis=1)
X_train = X_train[mask]
y_train = y_train[mask]

# 首先，确保 X_train 和 X_test 没有 NaN 值
X_train = X_train[~np.isnan(X_train).any(axis=1)]
y_train = y_train[~np.isnan(X_train).any(axis=1)]

class TimeseriesGenerator(Sequence):
    def __init__(self, data, labels, length, stride=1, batch_size=32):
        self.data = data
        self.labels = labels
        self.length = length
        self.stride = stride
        self.batch_size = batch_size

    def __len__(self):
        return max(int(np.ceil((len(self.data) - self.length) / float(self.stride * self.batch_size))), 0)

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        start = idx * self.batch_size * self.stride
        end = start + self.batch_size * self.stride + self.length

        for i in range(start, min(end, len(self.data) - self.length), self.stride):
            batch_x.append(self.data[i: i + self.length])
            batch_y.append(self.labels[i + self.length])

        return np.array(batch_x), np.array(batch_y)

# 定义时间步长和步长
time_steps = 100
stride = 1  # 增加步长以减少内存使用


model = load_model('model_taobao_lstm_limit_order.h5')


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


logger = DualWriter('tianqin_simu.log')

# 保存原始的stdout
original_stdout = sys.stdout

# 重定向stdout到我们的DualWriter
sys.stdout = logger


future_code = "SHFE.ss2403"
sim = TqSim(init_balance=10000)
sim.set_commission(future_code, 2)
# api = TqApi(sim,auth=TqAuth("卡卡罗特2023", "Hello2023"))
api = TqApi(TqAccount("H徽商期货", "952522", "Hello2023"), auth=TqAuth("卡卡罗特2023", "Hello2023"))
# 获得 i2209 tick序列的引用
ticks = api.get_tick_serial(future_code)
quote = api.get_quote(future_code)

def predict_next_move(tick, model, time_steps,historical_data,scaler):

    # 检查是否有足够的数据来计算滚动和EWM特征
    if len(historical_data) >= time_steps+10:

        for feature in pre_features:
            historical_data[feature + '_diff'] = historical_data[feature].diff()

        historical_data['last_price_diff'] = historical_data['last_price'].diff()
        historical_data['last_price_bid_diff'] =  historical_data['last_price'] - historical_data['bid_price1']  
        historical_data['last_price_ask_diff'] =  historical_data['last_price'] - historical_data['ask_price1']  
        historical_data['last_price_highest_diff'] =  historical_data['last_price'] - historical_data['highest']  
        historical_data['last_price_lowest_diff'] =  historical_data['last_price'] - historical_data['lowest']  
        

        data_for_scaling = historical_data[features].dropna()

        # 选择最近的time_steps行用于归一化
        data_to_scale = data_for_scaling.tail(time_steps)

        # 归一化
        scaled_data = scaler.fit_transform(data_to_scale)

        # 使用归一化的数据创建模型输入
        X_new = scaled_data.reshape(1, time_steps, len(features))


        # 检查X_new是否包含NaN值
        if np.isnan(X_new).any():
            return None, historical_data
        else:
            # 进行预测
            prediction_proba = model.predict(X_new,verbose=0)
            probability_of_one = prediction_proba[0][0]

            return probability_of_one, historical_data
    else:
        # 数据不足以进行预测
        return None, historical_data




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
notrade_time = ["11:20-11:30","14:50-15:00","1:30-2:00"]
# Initialize historical_data with the correct column names and types if necessary
historical_data = pd.DataFrame()
tick_count = 0
last_datetime = None  # 用于存储上次循环中最后一个tick的时间戳
last_buy_price = 0
lock = 1 
trade_count = 0
buy_count = 0
hold = 0
up_count = 0
all_count = 0

while True:
    api.wait_update()
    # 判断整个tick序列是否有变化
    if api.is_changing(ticks) and lock>0:
        lock = lock -1 
        new_ticks = ticks
        if last_datetime is not None:
            # 如果不是第一次循环，筛选出新的ticks
            new_ticks = ticks[ticks['datetime'] > last_datetime]
        for ind, new_tick in new_ticks.iterrows():
            tick_count+=1
            historical_data = pd.concat([historical_data, pd.DataFrame([new_tick])], ignore_index=True)

        tick = ticks.iloc[-1].to_dict()
        last_datetime = tick['datetime']
        #时段末尾不交易
        
        if is_time_in_ranges(datetime.fromtimestamp(last_datetime/1_000_000_000).time(),notrade_time):
            continue
        probability, historical_data = predict_next_move(tick, model, time_steps, historical_data,scaler)
        if probability is not None:
            # print(probability)
            buy_threshold = 0.6
            sold_threshold = 0.5
            account = api.get_account()
            position = api.get_position(future_code)
            # if position.pos_long==0 and probability>buy_threshold:
            if hold==0 and probability>buy_threshold:
                print(probability)
                print("buy:"+str(tick['bid_price1']))
                print(datetime.now())
                last_buy_price = tick['bid_price1']
                tick_count = 0
                hold=1
                price = tick['bid_price1']*quote['volume_multiple']*0.13
                sim.set_margin(future_code, price)
                volume = account.available // price
                if volume > 0:
                    order = api.insert_order(symbol=future_code, direction="BUY", offset="OPEN", limit_price=tick['bid_price1'], volume=volume)
                    start_time = datetime.now()
                    while True:
                        api.wait_update()
                        end_time = datetime.now()
                        if position.pos_long == volume:
                            tick_count = 0
                            buy_count = 0
                            print("buyed:"+str(tick['bid_price1']))
                            last_buy_price = tick['bid_price1']
                            break
                        elif (end_time - start_time).total_seconds() > 30:
                            api.cancel_order(order)
                            hold = 0
                            break
                else:
                    hold = 0
                        
            elif position.pos_long >0 and (tick['bid_price1']>last_buy_price or tick_count>100):
            # elif hold==1 and tick_count>100 and probability<sold_threshold:
                print(probability)
                print("sell:"+str(tick['bid_price1']))
                print(datetime.now())
                all_count += 1
                if tick['bid_price1']>last_buy_price:
                    up_count += 1
                print("diff:"+str(tick['bid_price1']-last_buy_price))
                print(all_count)
                print(up_count)
                order = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=tick['bid_price1'], volume=position.pos_long)
                start_time = datetime.now()
                while True:
                    api.wait_update()
                    end_time = datetime.now()
                    if position.pos_long == 0:
                        hold = 0
                        print("账户权益:%f, 账户余额:%f,持仓:%f" % (account.balance, account.available,position.pos_long))    
                        trade_count = trade_count+1
                        break
                    elif (end_time - start_time).total_seconds() > 10:
                        api.cancel_order(order)
                        break
        lock = lock + 1