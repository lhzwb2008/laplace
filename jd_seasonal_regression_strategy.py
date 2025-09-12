#!/usr/bin/env python3
"""
鸡蛋期货季节性回归策略
基于6-8月低点和11-1月高点的季节性规律，使用线性回归预测价格趋势
避免频繁换仓，提高策略效率
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class JDSeasonalRegressionStrategy:
    def __init__(self, initial_capital=20000, config=None):
        """
        初始化策略参数
        
        Args:
            initial_capital: 初始资金
            config: 策略配置参数
        """
        # 资金管理
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.margin_rate = 0.1  # 保证金率15%
        self.contract_multiplier = 10  # 合约乘数
        self.transaction_cost = 0.0001  # 手续费率
        self.slippage = 0.0005  # 滑点
        
        # 默认配置
        default_config = {
            'open_threshold': 0.05,  # 开仓阈值：偏离回归线8%
            'close_threshold': 0.01,  # 平仓阈值：回归到3%以内
            'stop_loss': 0.05,  # 止损：亏损10%
            'max_position_ratio': 0.5,  # 最大仓位比例
            'enable_trend_filter': True,  # 启用趋势过滤（顺大势逆小势）
        }
        
        # 合并用户配置
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # 策略参数
        self.low_months = [6, 7, 8]  # 低价月份（大势向上期间）
        self.high_months = [11, 12, 1]  # 高价月份（大势向下期间）
        self.open_threshold = self.config['open_threshold']
        self.close_threshold = self.config['close_threshold']
        self.stop_loss = self.config['stop_loss']
        self.max_position_ratio = self.config['max_position_ratio']
        self.enable_trend_filter = self.config['enable_trend_filter']
        self.min_days_to_delivery = 30  # 距离交割最少天数
        
        # 交易状态
        self.position = 0
        self.position_contract = None
        self.entry_price = 0
        self.entry_date = None
        
        # 记录
        self.trades = []
        self.daily_pnl = []
        self.chart_data = []  # 用于绘图的数据
        
    def get_contract_delivery_date(self, contract_code):
        """
        获取合约交割日期（假设每月15日交割）
        
        Args:
            contract_code: 合约代码，如'JD2510'
            
        Returns:
            datetime: 交割日期
        """
        year = int('20' + contract_code[2:4])
        month = int(contract_code[4:6])
        return datetime(year, month, 15)
    
    def get_days_to_delivery(self, contract_code, current_date):
        """
        计算距离交割的天数
        
        Args:
            contract_code: 合约代码
            current_date: 当前日期
            
        Returns:
            int: 距离交割的天数
        """
        delivery_date = self.get_contract_delivery_date(contract_code)
        return (delivery_date - current_date).days
    
    def get_theoretical_main_contract(self, current_date):
        """
        根据期货市场规律确定理论主力合约
        规则：距离交割60天成为主力，距离交割30天失去主力地位
        
        Args:
            current_date: 当前日期
            
        Returns:
            str: 理论主力合约代码
        """
        # 生成未来几个月的合约
        contracts = []
        year = current_date.year
        month = current_date.month
        
        # 生成接下来12个月的合约
        for i in range(12):
            target_month = month + i
            target_year = year
            
            if target_month > 12:
                target_month -= 12
                target_year += 1
                
            year_suffix = str(target_year)[-2:]
            contract_code = f"JD{year_suffix}{target_month:02d}"
            
            # 计算交割日期和距离交割天数
            delivery_date = self.get_contract_delivery_date(contract_code)
            days_to_delivery = (delivery_date - current_date).days
            
            # 主力合约：距离交割30-90天之间
            if 30 <= days_to_delivery <= 90:
                contracts.append((contract_code, days_to_delivery))
        
        if contracts:
            # 选择距离交割60天左右的合约（最接近60天的）
            contracts.sort(key=lambda x: abs(x[1] - 60))
            return contracts[0][0]
        
        return None
    
    def select_trading_contract(self, df, current_date):
        """
        选择交易合约
        基于期货市场规律选择主力合约
        
        Args:
            df: 当日数据
            current_date: 当前日期
            
        Returns:
            str: 选择的合约代码
        """
        # 首先确定理论主力合约
        theoretical_main = self.get_theoretical_main_contract(current_date)
        
        if theoretical_main is None:
            print(f"  ❌ 无法确定理论主力合约")
            return None
            
        # 检查理论主力合约是否有数据
        if theoretical_main in df['contract'].values:
            days_to_delivery = self.get_days_to_delivery(theoretical_main, current_date)
            return theoretical_main
        
        # 如果理论主力合约没有数据，按成交量选择
        print(f"  ⚠️ 理论主力合约{theoretical_main}无数据，按成交量选择")
        
        volume_sorted = df.groupby('contract')['volume'].sum().sort_values(ascending=False)
        if len(volume_sorted) == 0:
            return None
            
        # 从成交量最大的合约中选择距离交割合适的
        for contract in volume_sorted.index:
            days_to_delivery = self.get_days_to_delivery(contract, current_date)
            
            # 距离交割15-120天之间的合约可以交易
            if 15 <= days_to_delivery <= 120:
                contract_type = "成交量最大" if contract == volume_sorted.index[0] else "备选"
                print(f"  🎯 选择{contract_type}合约{contract}开仓（距离交割{days_to_delivery}天）")
                return contract
        
        print(f"  ❌ 所有合约都不适合交易")
        return None
    
    def get_last_year_prices(self, df, current_date):
        """
        获取去年夏天和冬天的平均价格
        
        Args:
            df: 历史数据
            current_date: 当前日期
            
        Returns:
            tuple: (去年夏天均价, 去年冬天均价)
        """
        # 确定"去年"的时间范围
        # 去年夏天：上一个6-8月
        # 去年冬天：上一个11-1月
        
        current_year = current_date.year
        current_month = current_date.month
        
        # 确定去年夏天的年份
        if current_month >= 9:  # 9月及以后，去年夏天就是今年的6-8月
            summer_year = current_year
        else:  # 9月之前，去年夏天是去年的6-8月
            summer_year = current_year - 1
            
        # 确定去年冬天的年份
        if current_month >= 2:  # 2月及以后，去年冬天是去年11月到今年1月
            winter_year = current_year - 1
        else:  # 1月，去年冬天是前年11月到去年1月
            winter_year = current_year - 2
            
        # 获取去年夏天数据（6-8月）
        summer_start = datetime(summer_year, 6, 1)
        summer_end = datetime(summer_year, 8, 31)
        summer_mask = (df['date'] >= summer_start) & (df['date'] <= summer_end)
        summer_data = df[summer_mask]
        
        # 获取去年冬天数据（11-1月）
        winter_start = datetime(winter_year, 11, 1)
        winter_end = datetime(winter_year + 1, 1, 31)
        winter_mask = (df['date'] >= winter_start) & (df['date'] <= winter_end)
        winter_data = df[winter_mask]
        
        # 计算均价
        summer_avg = summer_data['close'].mean() if len(summer_data) > 0 else None
        winter_avg = winter_data['close'].mean() if len(winter_data) > 0 else None
        
        return summer_avg, winter_avg
    
    def calculate_price_adjustment(self, df, current_date, last_summer_avg, last_winter_avg):
        """
        根据今年的数据计算价格调整系数
        
        Args:
            df: 历史数据
            current_date: 当前日期
            last_summer_avg: 去年夏天均价
            last_winter_avg: 去年冬天均价
            
        Returns:
            float: 价格调整系数（如1.1表示今年比去年高10%）
        """
        current_year = current_date.year
        current_month = current_date.month
        
        # 获取今年已有数据的均价
        year_start = datetime(current_year, 1, 1)
        year_mask = (df['date'] >= year_start) & (df['date'] <= current_date)
        year_data = df[year_mask]
        
        if len(year_data) == 0:
            return 1.0
            
        # 计算今年均价
        current_avg = year_data['close'].mean()
        
        # 计算去年同期均价（简单平均）
        last_year_avg = (last_summer_avg + last_winter_avg) / 2 if last_summer_avg and last_winter_avg else None
        
        if last_year_avg:
            # 计算调整系数
            adjustment = current_avg / last_year_avg
            # 限制调整幅度在0.8-1.2之间
            return max(0.8, min(1.2, adjustment))
        
        return 1.0
    
    def calculate_seasonal_trend_line(self, df, current_date):
        """
        计算季节性价格趋势线
        
        使用简化的方法：基于当前年份的实际数据计算季节性趋势
        
        Args:
            df: 历史数据
            current_date: 当前日期
            
        Returns:
            float: 当日的季节性趋势价格
        """
        current_year = current_date.year
        current_month = current_date.month
        
        # 获取当前年份的数据
        year_data = df[df['date'].dt.year == current_year]
        if len(year_data) == 0:
            return None
        
        # 如果当前年份数据不足，使用前一年数据
        if len(year_data) < 100:  # 数据太少
            year_data = df[df['date'].dt.year == current_year - 1]
            if len(year_data) == 0:
                return None
        
        # 计算当前年份的夏季和冬季均价
        summer_data = year_data[year_data['date'].dt.month.isin([6, 7, 8])]
        winter_data = year_data[year_data['date'].dt.month.isin([11, 12, 1])]
        
        if len(summer_data) == 0 or len(winter_data) == 0:
            # 如果当年数据不完整，使用历史均价
            return df['close'].mean()
        
        summer_avg = summer_data['close'].mean()
        winter_avg = winter_data['close'].mean()
        
        # 使用简单的正弦波模型模拟季节性变化
        # 7月为最低点 (month=7)，12月为最高点 (month=12)
        import numpy as np
        
        # 将月份转换为角度 (7月=0, 12月=π)
        if current_month >= 7:
            # 7-12月：上升期
            angle = np.pi * (current_month - 7) / 5  # 7月到12月，5个月
        else:
            # 1-6月：下降期 (从上年12月到当年7月)
            angle = np.pi * (current_month + 5) / 12  # 继续上升到1月，然后下降
        
        # 计算趋势价格 (在夏季均价和冬季均价之间变化)
        price_range = winter_avg - summer_avg
        trend_price = summer_avg + price_range * (1 + np.sin(angle - np.pi/2)) / 2
        
        return trend_price
    
    def is_uptrend_season(self, current_date):
        """
        判断当前是否处于上涨趋势季节（6-10月）
        
        Args:
            current_date: 当前日期
            
        Returns:
            bool: True表示上涨趋势季节
        """
        month = current_date.month
        return 6 <= month <= 10
    
    def generate_signal(self, current_price, trend_line, current_date):
        """
        基于季节性趋势线生成交易信号
        
        Args:
            current_price: 当前价格
            trend_line: 季节性趋势线价格
            current_date: 当前日期
            
        Returns:
            int: 1=做多, -1=做空, 0=平仓信号, None=保持现状
        """
        if trend_line is None:
            return None
            
        is_uptrend = self.is_uptrend_season(current_date)
        
        # 计算价格相对于趋势线的偏离度
        deviation = (current_price - trend_line) / trend_line
        
        if self.position == 0:
            # 无持仓，检查开仓信号
            
            # 价格显著低于趋势线，做多（价格被低估）
            if deviation < -self.open_threshold:
                if self.enable_trend_filter:
                    # 下跌季节（11-5月）不做多
                    if not is_uptrend:
                        return None
                return 1
                
            # 价格显著高于趋势线，做空（价格被高估）
            if deviation > self.open_threshold:
                if self.enable_trend_filter:
                    # 上涨季节（6-10月）不做空
                    if is_uptrend:
                        return None
                return -1
                
        else:
            # 有持仓，检查平仓信号
            if self.position > 0:  # 多头持仓
                # 价格回归到趋势线附近时平仓
                if deviation > -self.close_threshold:
                    return 0
                    
                # 止损检查：实际盈亏
                pnl_rate = (current_price - self.entry_price) / self.entry_price
                if pnl_rate < -self.stop_loss:
                    return 0
                    
            else:  # 空头持仓
                # 价格回归到趋势线附近时平仓
                if deviation < self.close_threshold:
                    return 0
                    
                # 止损检查：实际盈亏
                pnl_rate = (self.entry_price - current_price) / self.entry_price
                if pnl_rate < -self.stop_loss:
                    return 0
                    
        return None  # 保持现状
    
    def calculate_position_size(self, price):
        """
        计算仓位大小
        
        Args:
            price: 当前价格
            
        Returns:
            int: 仓位手数
        """
        # 计算可用资金
        available_capital = self.capital * self.max_position_ratio
        
        # 计算单手保证金
        margin_per_lot = price * self.contract_multiplier * self.margin_rate
        
        # 计算最大可开手数
        max_lots = int(available_capital / margin_per_lot)
        
        # 限制最大手数
        return min(max_lots, 3)  # 最多3手
    
    def open_position(self, signal, price, contract, date):
        """
        开仓
        
        Args:
            signal: 交易信号
            price: 开仓价格
            contract: 合约代码
            date: 交易日期
        """
        position_size = self.calculate_position_size(price)
        if position_size == 0:
            return
            
        # 计算成本
        trade_cost = price * position_size * self.contract_multiplier * (self.transaction_cost + self.slippage)
        
        # 更新持仓
        self.position = position_size * signal
        self.position_contract = contract
        self.entry_price = price
        self.entry_date = date
        
        # 扣除手续费
        self.capital -= trade_cost
        
        # 记录交易
        self.trades.append({
            'date': date,
            'action': '开仓',
            'contract': contract,
            'direction': '多头' if signal == 1 else '空头',
            'price': price,
            'position': self.position,
            'cost': trade_cost
        })
        
        days_to_delivery = self.get_days_to_delivery(contract, date)
        print(f"🎯 选择理论主力合约{contract}开仓（距离交割{days_to_delivery}天）")
        print(f"📊 开仓 [{date.strftime('%Y-%m-%d')}]: "
              f"合约{contract}, 方向{'多头' if signal == 1 else '空头'}, "
              f"价格{price:.0f}, 仓位{abs(self.position)}手")
    
    def close_position(self, price, date, reason):
        """
        平仓
        
        Args:
            price: 平仓价格
            date: 交易日期
            reason: 平仓原因
        """
        if self.position == 0:
            return
            
        # 计算盈亏
        if self.position > 0:
            pnl = (price - self.entry_price) * abs(self.position) * self.contract_multiplier
        else:
            pnl = (self.entry_price - price) * abs(self.position) * self.contract_multiplier
            
        # 计算手续费
        trade_cost = price * abs(self.position) * self.contract_multiplier * (self.transaction_cost + self.slippage)
        
        # 更新资金
        self.capital += pnl - trade_cost
        
        # 记录交易
        self.trades.append({
            'date': date,
            'action': '平仓',
            'contract': self.position_contract,
            'price': price,
            'pnl': pnl,
            'cost': trade_cost,
            'reason': reason
        })
        
        print(f"📊 平仓 [{date.strftime('%Y-%m-%d')}]: "
              f"合约{self.position_contract}, 价格{price:.0f}, "
              f"盈亏{pnl:.0f}元, 原因:{reason}")
        
        # 重置持仓
        self.position = 0
        self.position_contract = None
        self.entry_price = 0
        self.entry_date = None
    
    def preprocess_data(self, df):
        """
        预处理数据
        
        Args:
            df: 原始数据
            
        Returns:
            DataFrame: 处理后的日线数据
        """
        # 确保datetime列
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        df['date'] = df['datetime'].dt.date
        
        # 聚合成日线数据（所有合约）
        daily_data = df.groupby(['date', 'contract']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        
        return daily_data
    
    def backtest(self, df, start_date=None):
        """
        执行回测
        
        Args:
            df: 历史数据
            start_date: 回测开始日期，如'2021-01-01'，None表示使用数据最早日期
            
        Returns:
            DataFrame: 回测结果
        """
        # 预处理数据
        daily_data = self.preprocess_data(df)
        
        # 获取所有交易日
        all_dates = sorted(daily_data['date'].unique())
        
        # 过滤回测起始日期
        if start_date:
            start_date = pd.to_datetime(start_date)
            dates = [d for d in all_dates if d >= start_date]
            if len(dates) == 0:
                print(f"错误：指定的开始日期 {start_date.strftime('%Y-%m-%d')} 超出数据范围")
                return pd.DataFrame()
        else:
            dates = all_dates
        
        print(f"\n📈 执行回测（初始资金: {self.initial_capital:,}元）")
        print(f"数据范围: {all_dates[0].strftime('%Y-%m-%d')} 到 {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"回测范围: {dates[0].strftime('%Y-%m-%d')} 到 {dates[-1].strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        results = []
        
        for date in dates:
            # 获取当日数据
            day_data = daily_data[daily_data['date'] == date]
            
            if len(day_data) == 0:
                continue
            
            # 检查是否需要强制平仓（临近交割）
            if self.position != 0 and self.position_contract:
                days_to_delivery = self.get_days_to_delivery(self.position_contract, date)
                
                # 检查合约是否还有数据
                contract_data = day_data[day_data['contract'] == self.position_contract]
                
                if days_to_delivery < 15 or len(contract_data) == 0:
                    # 强制平仓（距离交割15天前必须平仓）
                    if len(contract_data) > 0:
                        close_price = contract_data.iloc[0]['close']
                    else:
                        # 使用主力合约价格估算
                        main_data = day_data.iloc[0]
                        close_price = main_data['close']
                    
                    self.close_position(close_price, date, 
                                      f"临近交割强制平仓(剩余{days_to_delivery}天)" if days_to_delivery < 15 
                                      else "合约无数据强制平仓")
                    continue
            
            # 计算季节性均价（使用主力合约数据）
            main_contract = day_data.groupby('contract')['volume'].sum().idxmax()
            main_data = day_data[day_data['contract'] == main_contract].iloc[0]
            current_price = main_data['close']
            
            # 获取历史主力合约数据用于计算均价
            historical_main = daily_data.groupby('date').apply(
                lambda x: x.loc[x['volume'].idxmax()]
            ).reset_index(drop=True)
            
            # 计算季节性趋势线
            trend_line = self.calculate_seasonal_trend_line(historical_main, date)
            
            if trend_line is None:
                continue
            
            # 生成交易信号
            signal = self.generate_signal(current_price, trend_line, date)
            
            # 收集绘图数据
            if self.position != 0:
                contract_data = day_data[day_data['contract'] == self.position_contract]
                if len(contract_data) > 0:
                    current_contract_price = contract_data.iloc[0]['close']
                    if self.position > 0:
                        unrealized_pnl = (current_contract_price - self.entry_price) * abs(self.position) * self.contract_multiplier
                    else:
                        unrealized_pnl = (self.entry_price - current_contract_price) * abs(self.position) * self.contract_multiplier
                else:
                    unrealized_pnl = 0
            else:
                unrealized_pnl = 0
                
            self.chart_data.append({
                'date': date,
                'price': current_price,
                'trend_line': trend_line,
                'total_value': self.capital + unrealized_pnl,
                'position': self.position
            })
            
            if signal is not None:
                if signal == 0 and self.position != 0:
                    # 平仓信号
                    self.close_position(current_price, date, "回归平仓")
                elif signal != 0 and self.position == 0:
                    # 开仓信号
                    trade_contract = self.select_trading_contract(day_data, date)
                    if trade_contract:
                        contract_data = day_data[day_data['contract'] == trade_contract].iloc[0]
                        self.open_position(signal, contract_data['close'], trade_contract, date)
            
            # 记录每日状态
            if self.position != 0:
                contract_data = day_data[day_data['contract'] == self.position_contract]
                if len(contract_data) > 0:
                    current_contract_price = contract_data.iloc[0]['close']
                    if self.position > 0:
                        unrealized_pnl = (current_contract_price - self.entry_price) * abs(self.position) * self.contract_multiplier
                    else:
                        unrealized_pnl = (self.entry_price - current_contract_price) * abs(self.position) * self.contract_multiplier
                else:
                    unrealized_pnl = 0
            else:
                unrealized_pnl = 0
            
            results.append({
                'date': date,
                'position': self.position,
                'contract': self.position_contract,
                'capital': self.capital,
                'unrealized_pnl': unrealized_pnl,
                'total_value': self.capital + unrealized_pnl
            })
        
        # 强制平仓所有持仓
        if self.position != 0:
            last_date = dates[-1]
            last_data = daily_data[daily_data['date'] == last_date]
            if self.position_contract:
                contract_data = last_data[last_data['contract'] == self.position_contract]
                if len(contract_data) > 0:
                    self.close_position(contract_data.iloc[0]['close'], last_date, "回测结束强制平仓")
        
        return pd.DataFrame(results)
    
    def print_summary(self):
        """
        打印策略总结
        """
        print("\n" + "=" * 60)
        print("📊 策略执行总结")
        print("=" * 60)
        
        # 资金情况
        print(f"\n💰 资金情况:")
        print(f"  初始资金: {self.initial_capital:,} 元")
        print(f"  最终资金: {self.capital:,.0f} 元")
        print(f"  总收益: {self.capital - self.initial_capital:,.0f} 元")
        print(f"  收益率: {(self.capital/self.initial_capital - 1)*100:.1f}%")
        
        # 交易统计
        print(f"\n📋 交易统计:")
        open_trades = [t for t in self.trades if t['action'] == '开仓']
        close_trades = [t for t in self.trades if t['action'] == '平仓']
        
        print(f"  开仓次数: {len(open_trades)}")
        print(f"  平仓次数: {len(close_trades)}")
        
        if close_trades:
            winning_trades = [t for t in close_trades if t['pnl'] > 0]
            print(f"  胜率: {len(winning_trades)/len(close_trades)*100:.1f}%")
            
            total_profit = sum(t['pnl'] for t in close_trades if t['pnl'] > 0)
            total_loss = sum(abs(t['pnl']) for t in close_trades if t['pnl'] < 0)
            if total_loss > 0:
                print(f"  盈亏比: {total_profit/total_loss:.2f}")
        
        # 无换仓统计
        print(f"\n✨ 策略特点:")
        print(f"  换仓次数: 0 (策略设计避免换仓)")
        print(f"  平均持仓时间: {self.calculate_avg_holding_days():.0f} 天")
        
    def calculate_avg_holding_days(self):
        """
        计算平均持仓天数
        
        Returns:
            float: 平均持仓天数
        """
        holding_days = []
        
        open_date = None
        for trade in self.trades:
            if trade['action'] == '开仓':
                open_date = trade['date']
            elif trade['action'] == '平仓' and open_date:
                days = (trade['date'] - open_date).days
                holding_days.append(days)
                open_date = None
        
        return np.mean(holding_days) if holding_days else 0
    
    def plot_strategy_chart(self, save_path='jd_strategy_chart.png'):
        """
        绘制策略图表：价格、回归线、交易点位
        
        Args:
            save_path: 图片保存路径
        """
        if not self.chart_data:
            print("没有绘图数据")
            return
            
        df_chart = pd.DataFrame(self.chart_data)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 上图：价格和季节性趋势线
        ax1.plot(df_chart['date'], df_chart['price'], 'b-', label='鸡蛋价格', linewidth=1.5)
        ax1.plot(df_chart['date'], df_chart['trend_line'], 'r-', label='季节性趋势线', linewidth=2, alpha=0.8)
        
        # 填充价格与趋势线之间的偏差区域
        ax1.fill_between(df_chart['date'], df_chart['price'], df_chart['trend_line'], 
                         alpha=0.2, color='gray', label='价格偏差')
        
        # 标记交易点位
        long_open_labeled = False
        short_open_labeled = False
        close_labeled = False
        
        for trade in self.trades:
            if trade['action'] == '开仓':
                color = 'red' if trade['direction'] == '多头' else 'blue'
                marker = '^' if trade['direction'] == '多头' else 'v'
                
                # 只为第一次出现的多头和空头开仓添加标签
                if trade['direction'] == '多头' and not long_open_labeled:
                    label = '多头开仓'
                    long_open_labeled = True
                elif trade['direction'] == '空头' and not short_open_labeled:
                    label = '空头开仓'
                    short_open_labeled = True
                else:
                    label = ""
                    
                ax1.scatter(trade['date'], trade['price'], color=color, marker=marker, s=100, label=label)
                
            elif trade['action'] == '平仓':
                label = '平仓' if not close_labeled else ""
                if not close_labeled:
                    close_labeled = True
                ax1.scatter(trade['date'], trade['price'], color='black', marker='x', s=100, alpha=0.8, label=label)
        
        ax1.set_title('鸡蛋期货季节性趋势线策略 - 价格与趋势线', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格 (元/吨)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 下图：资金曲线
        if len(df_chart) > 0:
            ax2.plot(df_chart['date'], df_chart['total_value'], 'g-', linewidth=2, label='总资产')
            ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='初始资金')
            
            # 标记交易点的资金变化
            for trade in self.trades:
                if trade['action'] == '平仓':
                    # 找到对应日期的资金值
                    trade_data = df_chart[df_chart['date'] <= trade['date']]
                    if len(trade_data) > 0:
                        latest_value = trade_data.iloc[-1]['total_value']
                        color = 'green' if trade['pnl'] > 0 else 'red'
                        ax2.scatter(trade['date'], latest_value, color=color, marker='o', s=50, alpha=0.8)
        
        ax2.set_title('资金曲线', fontsize=14, fontweight='bold')
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('资产价值 (元)', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 策略图表已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def run_strategy(self, data_file, start_date=None):
        """
        运行策略
        
        Args:
            data_file: 数据文件路径
            start_date: 回测开始日期，如'2021-01-01'
            
        Returns:
            DataFrame: 回测结果
        """
        # 读取数据
        print(f"📂 读取数据文件: {data_file}")
        df = pd.read_csv(data_file)
        print(f"数据量: {len(df)} 条")
        
        # 执行回测
        results = self.backtest(df, start_date)
        
        # 打印总结
        self.print_summary()
        
        return results


def main():
    """主函数"""
    
    # ==================== 策略配置区域 ====================
    
    # 🎯 核心策略参数 (可自由调整优化)
    strategy_config = {
        # 开平仓阈值
        'open_threshold': 0.15,      # 开仓阈值：价格偏离基准线多少比例开仓 (建议范围: 0.10-0.25)
        'close_threshold': 0.05,     # 平仓阈值：价格回归多少比例平仓 (建议范围: 0.02-0.08)
        
        # 风险控制
        'stop_loss': 0.15,           # 止损比例 (建议范围: 0.10-0.20)
        'max_position_ratio': 0.5,   # 最大仓位比例 (建议范围: 0.3-0.6)
        
        # 策略特性
        'enable_trend_filter': False, # 暂时关闭趋势过滤，看看基础策略效果
    }
    
    # 📊 回测配置
    config = {
        # 基础配置
        'initial_capital': 20000,    # 初始资金
        'data_file': 'jd_all_contracts_daily_2021-2025_20250912_154518.csv',  # 数据文件路径
        
        # 时间配置
        'start_date': '2021-01-01',  # 回测开始日期 (YYYY-MM-DD 或 None)
        
        # 输出配置
        'plot_chart': True,          # 是否生成策略图表
        'chart_filename': 'jd_strategy_chart.png',  # 图表文件名
    }
    
    # ==================== 配置区域结束 ====================
    
    # 创建策略实例
    strategy = JDSeasonalRegressionStrategy(config['initial_capital'], strategy_config)
    
    # 运行策略
    results = strategy.run_strategy(config['data_file'], config.get('start_date'))
    
    # 绘制策略图表
    if config.get('plot_chart', True):
        strategy.plot_strategy_chart(save_path=config.get('chart_filename', 'jd_strategy_chart.png'))
    
    # 计算性能指标
    if len(results) > 0:
        # 计算夏普比率
        results['daily_return'] = results['total_value'].pct_change()
        sharpe_ratio = results['daily_return'].mean() / results['daily_return'].std() * np.sqrt(252)
        
        # 计算最大回撤
        results['cummax'] = results['total_value'].cummax()
        results['drawdown'] = (results['total_value'] - results['cummax']) / results['cummax']
        max_drawdown = results['drawdown'].min()
        
        print(f"\n📈 风险指标:")
        print(f"  夏普比率: {sharpe_ratio:.2f}")
        print(f"  最大回撤: {max_drawdown*100:.1f}%")
    
    print("\n✅ 策略执行完成！")
    
    return strategy, results


if __name__ == "__main__":
    strategy, results = main()
