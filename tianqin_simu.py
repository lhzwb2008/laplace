#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

data_clean = pd.read_csv("future_ss2312_tick.csv")


# 1. 数据加载和预处理
# data_clean = data.sort_values(by='trade_time')
data_clean['close'] = pd.to_numeric(data_clean['close'], errors='coerce')
data_clean['trade_time'] = pd.to_datetime(data_clean['trade_time'])
data_clean['volume'] = pd.to_numeric(data_clean['数量'], errors='coerce')

# 2. 特征生成
# Example usage:
rolling_windows = {'mean': 300, 'std': 300, 'rsi': 400}
ewm_spans = {'short': 200, 'long': 800, 'signal': 800}

# 滚动统计特征
data_clean['rolling_mean'] = data_clean['close'].rolling(window=rolling_windows['mean']).mean()
data_clean['rolling_std'] = data_clean['close'].rolling(window=rolling_windows['std']).std()


# Calculate RSI
delta = data_clean['close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=rolling_windows['rsi']).mean()
avg_loss = loss.rolling(window=rolling_windows['rsi']).mean()
rs = avg_gain / avg_loss
data_clean['RSI'] = 100 - (100 / (1 + rs))

# Calculate MACD
short_ema = data_clean['close'].ewm(span=ewm_spans['short'], adjust=False).mean()
long_ema = data_clean['close'].ewm(span=ewm_spans['long'], adjust=False).mean()
data_clean['MACD'] = short_ema - long_ema
data_clean['MACD_signal'] = data_clean['MACD'].ewm(span=ewm_spans['signal'], adjust=False).mean()

# Shift RSI and MACD to use them as features for next timestep
data_clean['RSI_shifted'] = data_clean['RSI'].shift(1)
data_clean['MACD_shifted'] = data_clean['MACD'].shift(1)
data_clean['MACD_signal_shifted'] = data_clean['MACD_signal'].shift(1)

# 累积成交量特征
# 计算单位时间内的成交量变化
data_clean['volume_change'] = data_clean['volume'].diff()

# 累积成交量特征生成
data_clean['volume_rolling_mean'] = data_clean['volume'].rolling(window=300).mean()


# 价格变化率和成交量变化
data_clean['price_change_rate'] = data_clean['close'].pct_change()

# 定义标签
data_clean['label'] = (data_clean['close'].shift(-1000) > data_clean['close']).astype(int)


# 3. 分割数据
# 分割数据为训练集和测试集
# Convert the 'trade_time' column to datetime
data_clean['trade_time'] = pd.to_datetime(data_clean['trade_time'])

# Now you can filter the data between two dates
train_data = data_clean[(data_clean['trade_time'] >= '2023-09-20 09:00:00') & 
                        (data_clean['trade_time'] < '2023-10-15 09:00:00')]

test_data = data_clean[(data_clean['trade_time'] >= '2023-10-15 09:00:00') & 
                        (data_clean['trade_time'] < '2023-10-20 09:00:00')]


features = ['close', 'rolling_mean', 'rolling_std', 'RSI', 'MACD', 'volume_rolling_mean', 'price_change_rate', 'volume_change']


X_train = train_data[features].dropna()
y_train = train_data['label'][X_train.index]
X_test = test_data[features].dropna()
y_test = test_data['label'][X_test.index]

# 4. 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)# Use the cleaned data for training


#实时预测
import pandas as pd
from tqsdk import TqApi, TqAuth,TqSim,TargetPosTask
import datetime
import logging
import sys

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

# 使用方法：
# 创建一个DualWriter实例，指定日志文件名
logger = DualWriter('tianqin_simu.log')

# 保存原始的stdout
original_stdout = sys.stdout

# 重定向stdout到我们的DualWriter
sys.stdout = logger


sim = TqSim(init_balance=10000)
api = TqApi(sim,auth=TqAuth("卡卡罗特2023", "Hello2023"))

sim.set_commission("SHFE.ss2312", 2)
# 获得 i2209 tick序列的引用
ticks = api.get_tick_serial("SHFE.ss2312")



def predict_next_move(tick, model, rolling_windows, ewm_spans, historical_data):
    # 将 'last_price' 作为 'current' 进行计算
    tick['close'] = tick['last_price']
    
    # 将新的 tick 数据追加到历史数据中
    historical_data = pd.concat([historical_data, pd.DataFrame([tick])], ignore_index=True)
    
    # 打印当前已有的数据条数
    # print(f"当前已有的数据条数: {len(historical_data)}")
    
    # 检查我们是否有足够的数据来计算滚动和EWM特征
    if len(historical_data) >= max(rolling_windows['mean'], rolling_windows['std'], rolling_windows['rsi'], ewm_spans['long']):
        # 在历史数据上计算滚动平均和标准差
        historical_data['rolling_mean'] = historical_data['close'].rolling(window=rolling_windows['mean'], min_periods=1).mean()
        historical_data['rolling_std'] = historical_data['close'].rolling(window=rolling_windows['std'], min_periods=1).std()

        # 在历史数据上计算RSI
        delta = historical_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=rolling_windows['rsi'], min_periods=1).mean()
        avg_loss = loss.rolling(window=rolling_windows['rsi'], min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1)  # 避免除以零
        historical_data['RSI'] = 100 - (100 / (1 + rs))

        # 在历史数据上计算MACD
        short_ema = historical_data['close'].ewm(span=ewm_spans['short'], adjust=False, min_periods=1).mean()
        long_ema = historical_data['close'].ewm(span=ewm_spans['long'], adjust=False, min_periods=1).mean()
        historical_data['MACD'] = short_ema - long_ema
        historical_data['MACD_signal'] = historical_data['MACD'].ewm(span=ewm_spans['signal'], adjust=False, min_periods=1).mean()

        # 将RSI和MACD移位以用作下一个时间步的特征
        historical_data['RSI_shifted'] = historical_data['RSI'].shift(1)
        historical_data['MACD_shifted'] = historical_data['MACD'].shift(1)
        historical_data['MACD_signal_shifted'] = historical_data['MACD_signal'].shift(1)
        
         #累积成交量
        historical_data['volume_rolling_mean'] = historical_data['volume'].rolling(window=300).mean()

        # 价格变化率和成交量变化
        historical_data['price_change_rate'] = historical_data['close'].pct_change()
        historical_data['volume_change'] = historical_data['volume'].diff()

        # 使用最后一行数据进行预测
        X_new = historical_data.iloc[-1:][features]        
        # 检查X_new是否包含NaN值
        if X_new.isnull().values.any():
            # 处理包含NaN值的行（例如，跳过预测或使用占位符值）
            # 例如，我们可以返回None或一个特定的信号表示数据不足
            return None, historical_data
        else:
            # 进行预测
            prediction_proba = model.predict_proba(X_new)
            # 获取预测为类别1的概率
            probability_of_one = prediction_proba[0][1]
            return probability_of_one, historical_data
    else:
        # 数据不足以进行预测
        return None, historical_data


rolling_windows = {'mean': 300, 'std': 300, 'rsi': 400}
ewm_spans = {'short': 200, 'long': 800, 'signal': 800}


# Initialize historical_data with the correct column names and types if necessary
historical_data = pd.DataFrame()

tick_count = 0



while True:
    api.wait_update()
    # 判断整个tick序列是否有变化
    if api.is_changing(ticks):
        tick_count += 1
        tick = ticks.iloc[-1].to_dict()
        probability, historical_data = predict_next_move(tick, model, rolling_windows, ewm_spans, historical_data)
        if probability is not None:
            print(f"预测为1的概率: {probability}")
            buy_threshold = 0.9
            sold_threshold = 0.6
            account = api.get_account()
            position = api.get_position("SHFE.ss2312")
            quote = api.get_quote("SHFE.ss2312")
            target_pos = TargetPosTask(api, "SHFE.ss2312")
            if probability>buy_threshold and position.pos_long == 0:
                price = tick['last_price']*5*0.13
                sim.set_margin("SHFE.ss2312", price)
                volume = account.available // price
                if volume > 0:
                    target_pos.set_target_volume(volume)
                    while True:
                        api.wait_update()
                        if position.pos_long == volume:
                            tick_count = 0
                            with open('tianqin_simu.log', mode='a') as log:
                                log.write("buy:"+str(tick['last_price']))
                                log.write('\n') 
                                log.write(str(account))
                                log.write('\n') 
                                log.write(str(quote))
                                log.write('\n') 
                            break
                        
            elif position.pos_long >0 and tick_count>100 and probability<sold_threshold:
                target_pos.set_target_volume(0)
                while True:
                    if position.pos_long == 0:
                        api.wait_update()
                        with open('tianqin_simu.log', mode='a') as log:
                                log.write("sell:"+str(tick['last_price']))
                                log.write('\n') 
                                log.write(str(account))
                                log.write('\n') 
                                log.write(str(quote))
                                log.write('\n') 
                        break
            print(tick)
            print("账户权益:%f, 账户余额:%f,持仓:%f" % (account.balance, account.available,position.pos_long))       
        else:
            # print(tick)
            print("Insufficient data for prediction")

  