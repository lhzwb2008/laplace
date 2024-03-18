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
data = pd.read_csv(FILENAME+"_with_opportunities.csv")

data_clean = data.dropna().copy()  # 创建一个副本以避免警告

data_clean =  pd.get_dummies(data_clean, columns=['trade_color', 'trade_openclose'])

features = [
    'last_price','highest','lowest', 'volume', 'open_interest', 'volume_delta', 'open_interest_delta',
    'bid_price1', 'ask_price1', 'bid_price2', 'ask_price2', 'bid_price3', 'ask_price3',
    'bid_price4', 'ask_price4', 'bid_price5', 'ask_price5', 'bid_volume1', 'bid_volume2',
    'bid_volume3', 'bid_volume4', 'bid_volume5', 'ask_volume1', 'ask_volume2', 'ask_volume3',
    'ask_volume4', 'ask_volume5'
]

# 获取所有列名
all_columns = data_clean.columns.tolist()
# 筛选出以"trade_color_"和"trade_openclose_"开头的列名
# 并将它们添加到features列表中
for column in all_columns:
    if column.startswith('trade_color_') or column.startswith('trade_openclose_'):
        features.append(column)

window_size = 200  # 用于特征计算的窗口大小
# 特征生成
data_clean['rolling_mean'] = data_clean['last_price'].rolling(window=window_size).mean()
data_clean['rolling_std'] = data_clean['last_price'].rolling(window=window_size).std()
data_clean['ask_rolling_mean'] = data_clean['ask_price1'].rolling(window=window_size).mean()
data_clean['ask_rolling_std'] = data_clean['ask_price1'].rolling(window=window_size).std()
data_clean['bid_rolling_mean'] = data_clean['bid_price1'].rolling(window=window_size).mean()
data_clean['bid_rolling_std'] = data_clean['bid_price1'].rolling(window=window_size).std()
# RSI计算
delta = data_clean['last_price'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=window_size).mean()
avg_loss = loss.rolling(window=window_size).mean()
rs = avg_gain / avg_loss
data_clean['RSI'] = 100 - (100 / (1 + rs))
# MACD计算
short_ema = data_clean['last_price'].ewm(span=int(window_size/2), adjust=False).mean()  # 使用窗口大小的一半作为短期EMA
long_ema = data_clean['last_price'].ewm(span=window_size, adjust=False).mean()  # 使用整个窗口大小作为长期EMA
data_clean['MACD'] = short_ema - long_ema
data_clean['MACD_signal'] = data_clean['MACD'].ewm(span=int(window_size/3), adjust=False).mean()  # 使用窗口大小的三分之一作为信号线
# 删除因计算滚动特征而产生的NaN值
data_clean.dropna(inplace=True)
# 添加新特征到特征列表
features.extend(['rolling_mean', 'rolling_std','ask_rolling_mean', 'ask_rolling_std','bid_rolling_mean', 'bid_rolling_std', 'RSI', 'MACD', 'MACD_signal'])

X = data_clean[['datetime'] + features]
y = data_clean['dual_opportunity']

# 数据切分
split_start = int(len(data_clean) * 0.6)
split_point = int(len(data_clean) * 1)
split_end = int(len(data_clean) * 1)
X_train = X.iloc[split_start:split_point]
y_train = y.iloc[split_start:split_point]
X_test = X.iloc[split_point:split_end]
y_test = y.iloc[split_point:split_end]
X_train_features = X_train.drop(columns=['datetime'])
X_test_features = X_test.drop(columns=['datetime'])


# 创建XGBoost模型并调整参数
model = xgb.XGBClassifier(
    n_estimators=40,  # 树的个数
    max_depth=3,  # 树的深度
    learning_rate=0.1,  # 学习率
    subsample=0.8,  # 训练每棵树时使用的样本比例
    colsample_bytree=0.8,  # 构建树时的列采样比例
    random_state=42,  # 随机种子
    use_label_encoder=False,  # 避免使用标签编码器的警告
    eval_metric='logloss',  # 评估指标
    scale_pos_weight=1
)
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

def determine_trade_openclose_and_color(current_tick, previous_tick):
    # 默认值
    openclose = '未知'
    color = '白色'

    # 判断开平仓状态
    if current_tick['volume_delta'] == current_tick['open_interest_delta'] > 0:
        openclose = '双开'
    elif current_tick['volume_delta'] > current_tick['open_interest_delta'] > 0:
        openclose = '开仓'
    elif current_tick['volume_delta'] > abs(current_tick['open_interest_delta']) > 0 and current_tick['open_interest_delta'] < 0:
        openclose = '平仓'
    elif current_tick['volume_delta'] > 0 and current_tick['open_interest_delta'] == 0:
        openclose = '换手'
    elif current_tick['volume_delta'] + current_tick['open_interest_delta'] == 0 and current_tick['volume_delta'] > 0:
        openclose = '双平'
    elif current_tick['volume_delta'] == 0 and current_tick['open_interest_delta'] == 0:
        openclose = '无交易'
    
    # 判断价格方向来确定颜色
    if current_tick['last_price'] > previous_tick['last_price']:
        color = '红色'
    elif current_tick['last_price'] < previous_tick['last_price']:
        color = '绿色'
    else:
        # 进一步判断，如果价格没有变化，但是与买卖方报价有关，则可能为换手
        if current_tick['last_price'] >= current_tick['ask_price1'] or current_tick['last_price'] > previous_tick['ask_price1']:
            color = '红色'
        elif current_tick['last_price'] <= current_tick['bid_price1'] or current_tick['last_price'] < previous_tick['bid_price1']:
            color = '绿色'

    return openclose, color


def calculate_features(new_tick):
    global data_window
    # 将新的tick添加到data_window中
    data_window = pd.concat([data_window, pd.DataFrame([new_tick])], ignore_index=True)
    
    # 确保data_window不超过window_size指定的大小
    if len(data_window) > window_size:
        data_window = data_window.iloc[-window_size:]
    
    # 计算特征
    data_window['rolling_mean'] = data_window['last_price'].rolling(window=window_size).mean()
    data_window['rolling_std'] = data_window['last_price'].rolling(window=window_size).std()
    data_window['ask_rolling_mean'] = data_window['ask_price1'].rolling(window=window_size).mean()
    data_window['ask_rolling_std'] = data_window['ask_price1'].rolling(window=window_size).std()
    data_window['bid_rolling_mean'] = data_window['bid_price1'].rolling(window=window_size).mean()
    data_window['bid_rolling_std'] = data_window['bid_price1'].rolling(window=window_size).std()

    delta = data_window['last_price'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window_size).mean()
    avg_loss = loss.rolling(window=window_size).mean()
    rs = avg_gain / avg_loss
    data_window['RSI'] = 100 - (100 / (1 + rs))

    short_ema = data_window['last_price'].ewm(span=int(window_size/2), adjust=False).mean()
    long_ema = data_window['last_price'].ewm(span=window_size, adjust=False).mean()
    data_window['MACD'] = short_ema - long_ema
    data_window['MACD_signal'] = data_window['MACD'].ewm(span=int(window_size/3), adjust=False).mean()

    # 返回最新的tick附带计算好的特征
    return data_window.iloc[-1]

# 定义时间范围数组
notrade_time = ["11:25-11:30","14:55-15:00","0:50-1:00"]

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

# 假设data_window已经定义并初始化，例如：
data_window = pd.DataFrame(columns=features)

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

       
        window_size = 200  # 用于特征计算的窗口大小
        if len(data_window) < window_size:
            data_window = pd.concat([data_window, pd.DataFrame([tick])], ignore_index=True)
            continue
        tick = calculate_features(tick)
        previous_tick = data_window.iloc[-1] 
        
        # 调用函数计算trade_openclose和trade_color
        trade_openclose, trade_color = determine_trade_openclose_and_color(tick, previous_tick)
        # 将计算得到的trade_openclose和trade_color添加到current_tick中
        tick['trade_openclose'] = trade_openclose
        tick['trade_color'] = trade_color
        for feature in features:
            if feature.startswith('trade_color_') or feature.startswith('trade_openclose_'):
                tick[feature] = 0
        # 根据当前tick的 'trade_openclose' 和 'trade_color' 设置相应的字段为1
        tick[f"trade_openclose_{tick['trade_openclose']}"] = 1
        tick[f"trade_color_{tick['trade_color']}"] = 1
        
       
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
    
        
        if tick['ask_price1']-tick['bid_price1']!=trade_gap:
            continue
        
        # print(tick_features)
        probability = model.predict_proba(tick_features)[:, 1]
        # print(probability)
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
                long_price = tick['bid_price1']
                short_price = tick['ask_price1']
                order_buy = api.insert_order(symbol=future_code, direction="BUY", offset="OPEN", limit_price=tick['bid_price1'], volume=trade_hand)
                order_sell = api.insert_order(symbol=future_code, direction="SELL", offset="OPEN", limit_price=tick['ask_price1'], volume=trade_hand)
                start_time = datetime.now()
                tick_count = 0
                while True:
                    api.wait_update()
                    end_time = datetime.now()
                    count_ticks = api.get_tick_serial(future_code)
                    count_tick = count_ticks.iloc[-1]
                    if api.is_changing(count_ticks): 
                        tick_df = count_tick.to_frame().T
                        tick_df.to_csv('tianqin_ss.csv', mode='a', header=False, index=False)
                        
                        count_tick['volume_delta'] = count_tick['volume']-last_volume
                        last_volume = count_tick['volume']
                        if count_tick['volume_delta'] == 0 or count_tick['volume_delta']>1000:
                            continue
                        else:
                            tick_count += 1
                    else:
                        continue
                    if position.pos_long == trade_hand and position.pos_short == trade_hand:
                        print("open succeed")
                        lock = lock + 1
                        break
                    #提前止损
                    if position.pos_long == trade_hand and position.pos_short == 0 and count_tick['bid_price1']<long_price: 
                        print("sell open stoped")
                        api.cancel_order(order_sell)
                        order = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=count_tick['bid_price1'], volume=trade_hand) 
                        lock = lock + 1
                        break
                    if position.pos_long == 0 and position.pos_short == trade_hand and count_tick['ask_price1']>short_price: 
                        print("buy open stoped")
                        api.cancel_order(order_buy)
                        order = api.insert_order(symbol=future_code, direction="BUY", offset="CLOSETODAY", limit_price=count_tick['ask_price1'], volume=trade_hand) 
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
                long_price = tick['ask_price1']
                short_price = tick['bid_price1']
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
                    #提前止损
                    if position.pos_long == trade_hand and position.pos_short == 0 and count_tick['ask_price1']<long_price: 
                        print("sell close stoped")
                        api.cancel_order(order_sell)
                        while True :
                            api.wait_update()
                            if order_sell.status == "FINISHED":
                                break
                        order = api.insert_order(symbol=future_code, direction="SELL", offset="CLOSETODAY", limit_price=count_tick['bid_price1'], volume=trade_hand) 
                        lock = lock + 1
                        break
                    if position.pos_long == 0 and position.pos_short == trade_hand and count_tick['bid_price1']>short_price: 
                        print("buy close stoped")
                        api.cancel_order(order_buy)
                        while True :
                            api.wait_update()
                            if order_buy.status == "FINISHED":
                                break
                        order = api.insert_order(symbol=future_code, direction="BUY", offset="CLOSETODAY", limit_price=count_tick['ask_price1'], volume=trade_hand) 
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
                    
        