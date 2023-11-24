#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense,GRU,Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from keras.utils import Sequence
import datetime
import warnings
warnings.filterwarnings('ignore')

data_clean = pd.read_csv("future_ssmain_tick.csv")


# 1. 数据加载和预处理
# data_clean = data.sort_values(by='trade_time')
# 确保'close'列是数值型
data_clean['close'] = pd.to_numeric(data_clean['close'], errors='coerce')


price_features = ['昨收盘', '今开盘', '最高价', '最低价', '申买价一', '申卖价一']

for feature in price_features:
    data_clean[feature + '_diff'] = data_clean['close'] - data_clean[feature]

data_clean['trade_time'] = pd.to_datetime(data_clean['trade_time'])

data_clean['close_diff'] = data_clean['close'].diff()

# Define label
data_clean['label'] = (data_clean['close'].shift(-100) > data_clean['close']).astype(int)

features = ['close_diff', '数量'] + [f + '_diff' for f in price_features]

# 3. 分割数据

# Now you can filter the data between two dates
train_data = data_clean[(data_clean['trade_time'] >= '2023-09-01 09:00:00') & 
                        (data_clean['trade_time'] < '2023-10-25 09:00:00')]

test_data = data_clean[(data_clean['trade_time'] >= '2023-10-25 09:00:00') & 
                        (data_clean['trade_time'] < '2023-10-31 09:00:00')]


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
        return int(np.ceil((len(self.data) - self.length) / float(self.stride * self.batch_size)))

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
time_steps = 300
stride = 1  # 增加步长以减少内存使用

from keras.models import load_model
model = load_model('model_lstm.h5')

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
quote = api.get_quote("SHFE.ss2312")

def predict_next_move(tick, model, time_steps,historical_data,scaler):

    # 检查是否有足够的数据来计算滚动和EWM特征
    if len(historical_data) >= time_steps+10:

        for feature in price_features:
            historical_data[feature + '_diff'] = historical_data['close'] - historical_data[feature]

        historical_data['close_diff'] = historical_data['close'].diff()

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


# Initialize historical_data with the correct column names and types if necessary
historical_data = pd.DataFrame()
tick_count = 0
last_datetime = None  # 用于存储上次循环中最后一个tick的时间戳
last_buy_price = 0
while True:
    api.wait_update()
    # 判断整个tick序列是否有变化
    if api.is_changing(ticks):
        new_ticks = ticks
        if last_datetime is not None:
            # 如果不是第一次循环，筛选出新的ticks
            new_ticks = ticks[ticks['datetime'] > last_datetime]
        for ind, new_tick in new_ticks.iterrows():
            tick_count+=1
            new_tick['close'] = new_tick['last_price']
            new_tick['昨收盘'] = quote['pre_close']
            new_tick['今开盘'] = quote['open']
            new_tick['最高价'] = new_tick['highest'] 
            new_tick['最低价'] = new_tick['lowest']
            new_tick['申买价一'] = new_tick['bid_price1']
            new_tick['申卖价一'] = new_tick['ask_price1']
            new_tick['数量'] = new_tick['volume']
            # 将新的 tick 数据追加到历史数据中
            historical_data = pd.concat([historical_data, pd.DataFrame([new_tick])], ignore_index=True)

        tick = ticks.iloc[-1].to_dict()
        last_datetime = tick['datetime']
        probability, historical_data = predict_next_move(tick, model, time_steps, historical_data,scaler)
        if probability is not None:
            # print(f"预测为1的概率: {probability}")
            # print(f"lastprice: {tick['last_price']}")
            # print(f"tick_count: {tick_count}")
            buy_threshold = 0.8
            sold_threshold = 0.4
            account = api.get_account()
            position = api.get_position("SHFE.ss2312")
            target_pos = TargetPosTask(api, "SHFE.ss2312",price="ACTIVE")
            if probability>buy_threshold:
                price = tick['last_price']*quote['volume_multiple']*0.13
                sim.set_margin("SHFE.ss2312", price)
                volume = account.available // price
                if volume > 0:
                    target_pos.set_target_volume(volume)
                    start_time = datetime.datetime.now()
                    while True:
                        api.wait_update()
                        end_time = datetime.datetime.now()
                        if position.pos_long == volume:
                            tick_count = 0
                            last_buy_price = tick['ask_price1']
                        elif (end_time - start_time).total_seconds() > 10:
                            # 取消 TargetPosTask 实例
                            target_pos.cancel()
                            while not target_pos.is_finished():  # 此循环等待 target_pos_passive 处理 cancel 结束
                                api.wait_update()    
                            target_pos = TargetPosTask(api, "SHFE.ss2312",price="PASSIVE")
                            break
                        
            elif position.pos_long >0 and tick_count>30 and probability<sold_threshold:
                target_pos.set_target_volume(0)
                start_time = datetime.datetime.now()
                while True:
                    api.wait_update()
                    end_time = datetime.datetime.now()
                    if position.pos_long == 0:
                        print("账户权益:%f, 账户余额:%f,持仓:%f" % (account.balance, account.available,position.pos_long))       
                        with open('tianqin_simu.log', mode='a') as log:
                                log.write("sell,price diff:"+str(tick['bid_price1']-last_buy_price))
                                log.write('\n') 
                        break
                    elif (end_time - start_time).total_seconds() > 10:
                        print("账户权益:%f, 账户余额:%f,持仓:%f" % (account.balance, account.available,position.pos_long))   
                        # 取消 TargetPosTask 实例
                        target_pos.cancel()
                        while not target_pos.is_finished():  # 此循环等待 target_pos_passive 处理 cancel 结束
                            api.wait_update()    
                        target_pos = TargetPosTask(api, "SHFE.ss2312",price="PASSIVE")
                        break