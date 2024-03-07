import pandas as pd
import xgboost as xgb
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,precision_score
from tqsdk import TqApi, TqAuth,TqSim,TargetPosTask,TqAccount
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')


# 加载数据
FILENAME = "future_taobao_ssMain_tick"
# data = pd.read_csv(FILENAME+"_with_opportunities.csv")
data = pd.read_csv(FILENAME+"_120_with_opportunities.csv")

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
split_start = int(len(data_clean) * 0.5)
split_point = int(len(data_clean) * 1)
X_train = X.iloc[split_start:split_point]
y_train = y.iloc[split_start:split_point]
X_test = X.iloc[split_point:]
y_test = y.iloc[split_point:]

# 模型训练
model = xgb.XGBClassifier(
    n_estimators=50,  # 树的个数
    max_depth=4,  # 树的深度
    learning_rate=0.1,  # 学习率
    subsample=0.8,  # 训练每棵树时使用的样本比例
    colsample_bytree=0.8,  # 构建树时的列采样比例
    random_state=42,  # 随机种子
    use_label_encoder=False,  # 避免使用标签编码器的警告
    eval_metric='logloss'  # 评估指标
)
X_train_features = X_train.drop(columns=['datetime'])
X_test_features = X_test.drop(columns=['datetime'])
# 模型训练
model.fit(X_train_features, y_train)



class DualWriter:
    def __init__(self, filename):
        self.file = open(filename, 'a')
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
original_stdout = sys.stdout
sys.stdout = logger
        
        
# 检查是否每个相邻tick的变化符合整体趋势方向
def check_trend_consistency(prices):
    # 计算整体趋势方向
    overall_trend_up = prices[-1] > prices[0]
    
    # 检查每个相邻tick的变化是否符合整体趋势方向
    for i in range(len(prices) - 1):
        if overall_trend_up:
            # 如果整体趋势向上，但发现任何相邻tick价格下降，则不一致
            if prices[i + 1] < prices[i]:
                return False
        else:
            # 如果整体趋势向下，但发现任何相邻tick价格上升，则不一致
            if prices[i + 1] > prices[i]:
                return False                
    # 如果所有相邻tick的变化都符合整体趋势方向，则返回True
    return True
        
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
notrade_time = ["09:00-09:20","11:20-11:30","13:30-13:40","14:50-15:00","21:00-21:10","0:00-1:00"]

future_code = "SHFE.ss2405"
# sim = TqSim(init_balance=20000)
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
trade_threshold = 0.65
trade_hand = 1
trade_gap = 5 #每跳多少钱
guess_tick = 120

jump_tick = 0
account = api.get_account()
init_balance = account.balance
while True:
    try:
        api.wait_update()
    except Exception as e:
        if "运维时间" in str(e):
            print("检测到维护时间，等待重试...")
            time.sleep(600) 
    # 判断整个tick序列是否有变化
    if api.is_changing(ticks): 
        tick = ticks.iloc[-1]
        tick_df = tick.to_frame().T
        file_name = 'tianqin_ss.csv'
        file_exists = os.path.exists(file_name)
        tick_df.to_csv(file_name, mode='a', header=not file_exists, index=False)
        
        
        if last_datetime is not None:
            last_datetime = tick['datetime']
            tick['volume_delta'] = tick['volume']-last_volume
            tick['open_interest_delta'] = tick['open_interest'] - last_open_interest
            last_volume = tick['volume']
            last_open_interest = tick['open_interest']
        else:
            last_datetime = tick['datetime']
            last_volume = tick['volume']
            last_open_interest = tick['open_interest']
            continue
        
        if tick['volume_delta']<= 0:
            continue

        if jump_tick>0:
            jump_tick -= 1
            continue
        
        # 将整个datetime列转换为秒级时间戳，并格式化为标准日期时间字符串
        tick['datetime'] = pd.to_datetime(tick['datetime'].astype(float), unit='ns')
        # 本地化为UTC时间，然后转换为目标时区，例如 'Asia/Shanghai' 为中国标准时间
        tick['datetime'] = tick['datetime'].tz_localize('UTC').tz_convert('Asia/Shanghai')
        # 将时间格式化为标准日期时间字符串，去除时区信息
        tick['datetime'] = tick['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        datetime_str = tick['datetime'] 
        date_str = datetime_str.split(' ')[0]  # 新增：提取日期字符串
        
        
        #不交易时段
        if is_time_in_ranges(datetime.fromtimestamp(last_datetime/1_000_000_000).time(),notrade_time):
            continue
        
        
        position = api.get_position(future_code)
        if position.pos_long > trade_hand or position.pos_short > trade_hand:
            print("exceed trade_hand")
            break
        
        tick_df = tick.to_frame().T
        tick_features = tick_df[features]
        for column in tick_features.columns:
            tick_features[column] = pd.to_numeric(tick_features[column], errors='coerce')
        
        #不要一边倒趋势
        data = pd.read_csv(file_name)
        latest_ticks = data.tail(50)
        if len(latest_ticks) < 50:
            continue
        # 获取ask_price1列的值
        ask_prices = latest_ticks['ask_price1'].values
        # 检查整体变化是否在15以内
        # print(abs(ask_prices[-1] - ask_prices[0]))
        if (check_trend_consistency(ask_prices) and abs(ask_prices[-1] - ask_prices[0]) > 5) or abs(ask_prices[-1] - ask_prices[0]) > 15:
            print("overall_change_without_limit，jump 100")
            jump_tick = 100
            continue
            
        
        if tick['ask_price1']-tick['bid_price1']!=trade_gap:
            continue
        
        probability = model.predict_proba(tick_features)[:, 1]
        if probability < trade_threshold:
            continue
        
        if lock > 0:
            print(account.balance)
            if account.balance-init_balance < -120:
                print("lost too much today,stop")
                break
            lock = lock - 1
            if position.pos_long == 0 and position.pos_short == 0:
                print("start open")
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
                        tick_df = count_tick.to_frame().T
                        tick_df.to_csv('tianqin_ss.csv', mode='a', header=False, index=False)
                        
                        count_tick['volume_delta'] = count_tick['volume']-last_volume
                        last_volume = count_tick['volume']
                        if count_tick['volume_delta'] == 0 or count_tick['volume_delta']>1000:
                            continue
                        else:
                            tick_count += 1
                    if position.pos_long == trade_hand and position.pos_short == trade_hand:
                        print("open succeed")
                        lock = lock + 1
                        break
                    if tick_count >= guess_tick:
                        if position.pos_long == trade_hand and position.pos_short == 0: 
                            print("sell open faild")
                            api.cancel_order(order_sell)
                            order = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=count_tick['bid_price1'], volume=trade_hand) 
                            lock = lock + 1
                            break
                        elif position.pos_long == 0 and position.pos_short == trade_hand:
                            print("buy open faild")
                            api.cancel_order(order_buy)
                            order = api.insert_order(symbol=future_code, direction="BUY", offset="CLOSETODAY", limit_price=count_tick['ask_price1'], volume=trade_hand) 
                            lock = lock + 1
                            break
                        else:
                            api.cancel_order(order_buy)
                            api.cancel_order(order_sell)
                            lock = lock + 1
                            break
            elif position.pos_long == trade_hand and position.pos_short == trade_hand:
                #双平
                print("start close")
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
                        tick_df = count_tick.to_frame().T
                        tick_df.to_csv('tianqin_ss.csv', mode='a', header=False, index=False)
                        
                        count_tick['volume_delta'] = count_tick['volume']-last_volume
                        last_volume = count_tick['volume']
                        if count_tick['volume_delta'] == 0 or count_tick['volume_delta']>1000:
                            continue
                        else:
                            tick_count += 1
                    if position.pos_long == 0 and position.pos_short == 0:
                        print("close succeed")
                        lock = lock + 1
                        break
                    if tick_count >= guess_tick:
                        if position.pos_long == trade_hand and position.pos_short == 0: 
                            print("sell close faild")
                            api.cancel_order(order_sell)
                            while True :
                                api.wait_update()
                                if order_sell.status == "FINISHED":
                                    break
                            order = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=count_tick['bid_price1'], volume=trade_hand) 
                            lock = lock + 1
                            break
                        elif position.pos_long == 0 and position.pos_short == trade_hand:
                            print("buy close faild")
                            api.cancel_order(order_buy)
                            while True :
                                api.wait_update()
                                if order_buy.status == "FINISHED":
                                    break
                            order = api.insert_order(symbol=future_code, direction="BUY", offset="CLOSETODAY", limit_price=count_tick['ask_price1'], volume=trade_hand) 
                            lock = lock + 1
                            break
                        else:
                            api.cancel_order(order_buy)
                            api.cancel_order(order_sell)
                            lock = lock + 1
                            break
                    
        