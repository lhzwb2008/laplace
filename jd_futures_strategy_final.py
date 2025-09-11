#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鸡蛋期货策略 - 完美换仓版本

完全修正换仓逻辑：
1. 基于固定的交割前30天启动监控
2. 监控主力次主力价差，寻找最佳换仓时机
3. 记录每一次换仓的详细过程
4. 目标：找到接近零价差的换仓时机

作者：Assistant
创建时间：2025-01-09
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class JDStrategyPerfectRollover:
    """鸡蛋期货策略 - 完美换仓版本"""
    
    def __init__(self, initial_capital=100000):
        # 策略参数
        self.buy_months = [6, 7, 8]  # 夏季买入
        self.sell_months = [11, 12, 1]  # 冬季卖出
        self.buy_threshold = 0.5
        self.sell_threshold = 0.5
        self.price_pos_period = 30
        self.rsi_period = 14
        self.rsi_buy_max = 75
        self.rsi_sell_min = 25
        self.vol_max = 3.0
        
        # 风险管理参数
        self.initial_capital = initial_capital
        self.margin_rate = 0.10
        self.max_position_ratio = 0.9
        self.risk_per_trade = 0.2
        self.transaction_cost = 0.0005
        self.slippage = 0.0002
        self.contract_multiplier = 10
        
        # 换仓参数（严格设置以大幅降低换仓成本）
        self.rollover_days_before_delivery = 30  # 交割前30天开始监控（更早开始寻找机会）
        self.max_acceptable_spread = 0.003  # 最大可接受价差0.3%（极其严格）
        self.target_spread = 0.001  # 目标价差0.1%（追求近乎零价差）
        self.min_monitoring_days = 10  # 最少监控10天（充分等待最佳时机）
        self.force_rollover_days = 3  # 距离交割3天内强制换仓（最后期限）
        
        # 交易状态
        self.capital = initial_capital
        self.position = 0
        self.position_contract = None
        self.position_value = 0
        self.margin_used = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.trades = []
        
        # 换仓记录
        self.rollover_records = []
        self.rollover_costs = 0
        self.spread_monitoring = {}  # 价差监控记录
        
        # 目标保证金占用率
        self.target_margin_usage = 0.6
        
    def get_contract_delivery_date(self, contract_code):
        """获取合约的交割日期（每月最后一个交易日，大约是月末）"""
        year = int('20' + contract_code[2:4])
        month = int(contract_code[4:6])
        
        # 获取下个月的第一天，然后减去一天得到本月最后一天
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year
        
        next_month_first = datetime(next_year, next_month, 1)
        delivery_date = next_month_first - timedelta(days=1)
        return delivery_date
    
    def get_next_contract(self, current_contract):
        """获取下一个月的合约"""
        year = int('20' + current_contract[2:4])
        month = int(current_contract[4:6])
        
        next_month = month + 1
        next_year = year
        if next_month > 12:
            next_month = 1
            next_year += 1
        
        return f"JD{next_year % 100:02d}{next_month:02d}"
    
    def should_start_rollover_monitoring(self, current_date, contract, df):
        """判断是否应该开始换仓监控（交割前一个月开始监控）"""
        if not contract:
            return False
        
        # 从合约名称解析年月：JD2510 -> 2025年10月
        year = int('20' + contract[2:4])
        month = int(contract[4:6])
        
        # 计算监控开始时间：交割前一个月的1号
        # 例如：JD2510（2025年10月交割）-> 2025年9月1日开始监控
        if month == 1:
            monitor_year = year - 1
            monitor_month = 12
        else:
            monitor_year = year
            monitor_month = month - 1
        
        monitor_start_date = datetime(monitor_year, monitor_month, 1)
        
        # 如果当前日期 >= 监控开始日期，就开始监控
        if current_date >= monitor_start_date:
            # 还需要检查下一个合约是否有数据
            next_contract = self.get_next_contract(contract)
            current_day_data = df[df['datetime'].dt.date == current_date.date()]
            next_contract_data = current_day_data[current_day_data['contract'] == next_contract]
            
            return len(next_contract_data) > 0
        
        return False
    
    def calculate_spread(self, df, date, old_contract, new_contract):
        """计算两个合约的价差"""
        day_data = df[df['datetime'].dt.date == date.date()]
        
        old_data = day_data[day_data['contract'] == old_contract]
        new_data = day_data[day_data['contract'] == new_contract]
        
        if len(old_data) == 0 or len(new_data) == 0:
            return None
        
        old_price = old_data.iloc[-1]['close']
        new_price = new_data.iloc[-1]['close']
        old_volume = old_data['volume'].sum()
        new_volume = new_data['volume'].sum()
        
        spread = (new_price - old_price) / old_price
        
        return {
            'date': date,
            'old_contract': old_contract,
            'new_contract': new_contract,
            'old_price': old_price,
            'new_price': new_price,
            'spread': spread,
            'old_volume': old_volume,
            'new_volume': new_volume,
            'abs_spread': abs(spread)
        }
    
    def should_execute_rollover(self, spread_data, monitoring_history):
        """判断是否应该执行换仓（优化版本 - 更严格的条件）"""
        if not spread_data:
            return False, ""
        
        current_spread = spread_data['spread']
        abs_spread = abs(current_spread)
        monitoring_days = len(monitoring_history)
        
        # 条件1：价差小于目标价差（0.1%），立即换仓
        if abs_spread <= self.target_spread:
            return True, f"价差{current_spread*100:.2f}%达到目标"
        
        # 条件2：价差小于最大可接受价差（0.3%），根据监控天数灵活决策
        if abs_spread <= self.max_acceptable_spread:
            if monitoring_days >= self.min_monitoring_days:
                # 监控10天以上，检查是否为最优时机
                recent_spreads = [h['abs_spread'] for h in monitoring_history]
                min_spread = min(recent_spreads)
                if abs_spread <= min_spread:
                    return True, f"价差{current_spread*100:.2f}%为{monitoring_days}天监控期最优"
        
        # 条件3：监控时间不足但价差很小时，也考虑换仓
        elif monitoring_days < self.min_monitoring_days and abs_spread <= self.target_spread:
            return True, f"价差{current_spread*100:.2f}%极小，提前换仓"
        
        # 条件4：新合约成交量明显更大（主力已切换），但价差必须合理
        total_volume = spread_data['old_volume'] + spread_data['new_volume']
        if total_volume > 0:
            new_volume_ratio = spread_data['new_volume'] / total_volume
            if new_volume_ratio > 0.95 and abs_spread <= self.max_acceptable_spread * 1.5:  # 放宽到0.45%
                return True, f"新合约成交量占比{new_volume_ratio*100:.1f}%，主力已切换"
        
        # 条件5：强制换仓（距离交割日3天内）
        delivery_date = self.get_contract_delivery_date(spread_data['old_contract'])
        days_to_delivery = (delivery_date - spread_data['date']).days
        if days_to_delivery <= self.force_rollover_days:
            return True, f"距离交割仅{days_to_delivery}天，强制换仓"
        
        # 条件6：监控时间过长（超过20天），选择相对最优时机
        if monitoring_days >= 20:
            all_spreads = [h['abs_spread'] for h in monitoring_history]
            percentile_20 = sorted(all_spreads)[int(len(all_spreads) * 0.2)]  # 20%分位数
            if abs_spread <= percentile_20:
                return True, f"监控{monitoring_days}天，价差{current_spread*100:.2f}%处于20%分位数"
        
        return False, f"价差{current_spread*100:.2f}%过高，继续等待（已监控{monitoring_days}天）"
    
    def execute_rollover(self, spread_data, reason):
        """执行换仓操作"""
        if self.position == 0:
            return
        
        old_price = spread_data['old_price']
        new_price = spread_data['new_price']
        old_contract = spread_data['old_contract']
        new_contract = spread_data['new_contract']
        
        # 计算换仓成本
        rollover_cost = self.calculate_rollover_cost(old_price, new_price, self.position)
        
        # 更新资金
        self.capital -= rollover_cost
        self.rollover_costs += rollover_cost
        
        # 记录换仓
        self.rollover_records.append({
            'date': spread_data['date'],
            'old_contract': old_contract,
            'new_contract': new_contract,
            'old_price': old_price,
            'new_price': new_price,
            'position': self.position,
            'cost': rollover_cost,
            'spread': spread_data['spread'] * 100,
            'reason': reason
        })
        
        # 更新持仓合约
        self.position_contract = new_contract
        
        # 更新入场价格（按照价差调整）
        price_ratio = new_price / old_price
        self.entry_price = self.entry_price * price_ratio
        self.stop_loss_price = self.stop_loss_price * price_ratio
        
        print(f"🔄 [{spread_data['date'].strftime('%Y-%m-%d')}] 换仓: {old_contract}({old_price:.0f})→{new_contract}({new_price:.0f}) 价差{spread_data['spread']*100:+.2f}% 成本{rollover_cost:.0f}元 | {reason}")
    
    def calculate_rollover_cost(self, old_price, new_price, position_size):
        """
        计算换仓成本
        
        换仓成本 = 手续费 + 滑点成本
        价差本身不是成本，因为：
        1. 换仓是为了维持相同的市场敞口
        2. 价差反映的是不同交割月份的合理定价差异
        3. 真正的成本只有交易摩擦成本
        """
        
        # 交易成本：平仓旧合约 + 开仓新合约的手续费和滑点
        transaction_cost = (old_price + new_price) * abs(position_size) * \
                          self.contract_multiplier * (self.transaction_cost + self.slippage)
        
        return transaction_cost
    
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_price_position(self, prices, period=30):
        """计算价格在指定周期内的分位数位置"""
        return prices.rolling(window=period).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
    
    def calculate_volatility(self, prices, period=20):
        """计算价格波动率"""
        returns = prices.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)
    
    def calculate_position_size(self, price, direction):
        """计算合理的仓位大小"""
        available_capital = self.capital - self.margin_used
        
        target_margin_amount = self.capital * self.target_margin_usage
        target_position = target_margin_amount / (price * self.margin_rate * 10)
        
        max_risk_amount = self.capital * self.risk_per_trade
        fixed_stop_loss_ratio = 0.10
        risk_based_position = max_risk_amount / (price * fixed_stop_loss_ratio * 10)
        
        safety_margin = 0.05
        safe_capital = available_capital * (1 - safety_margin)
        max_safe_position = safe_capital / (price * self.margin_rate * 10)
        
        position_size = min(target_position, max_safe_position, risk_based_position * 2.0)
        
        final_position = max(1, int(position_size))
        
        required_margin = price * final_position * 10 * self.margin_rate
        max_allowed_margin = self.capital * self.target_margin_usage
        if required_margin > max_allowed_margin:
            final_position = max(1, int(max_allowed_margin / (price * 10 * self.margin_rate)))
        
        return final_position
    
    def calculate_stop_loss(self, entry_price, direction):
        """计算止损价格"""
        stop_loss_ratio = 0.10
        if direction == 1:
            return entry_price * (1 - stop_loss_ratio)
        else:
            return entry_price * (1 + stop_loss_ratio)
    
    def execute_trade(self, signal, price, contract, reason, date):
        """执行交易"""
        if self.position != 0:
            self.close_position(price, date, "信号切换")
        
        if signal != 0:
            position_size = self.calculate_position_size(price, signal)
            
            trade_cost = price * position_size * 10 * (self.transaction_cost + self.slippage)
            required_margin = price * position_size * 10 * self.margin_rate
            
            available_capital = self.capital - self.margin_used
            
            if required_margin > available_capital:
                max_affordable_size = int(available_capital / (price * 10 * self.margin_rate))
                if max_affordable_size < 1:
                    return False
                position_size = max_affordable_size
                trade_cost = price * position_size * 10 * (self.transaction_cost + self.slippage)
                required_margin = price * position_size * 10 * self.margin_rate
            
            self.position = position_size * signal
            self.position_contract = contract
            self.entry_price = price
            self.stop_loss_price = self.calculate_stop_loss(price, signal)
            self.position_value = price * abs(self.position) * 10
            self.margin_used += required_margin
            
            self.trades.append({
                'date': date,
                'action': '开仓',
                'contract': contract,
                'direction': '多头' if signal == 1 else '空头',
                'price': price,
                'position': self.position,
                'margin': required_margin,
                'stop_loss': self.stop_loss_price,
                'entry_cost': trade_cost,
                'reason': reason
            })
            
            print(f"📊 开仓 [{date.strftime('%Y-%m-%d')}]: "
                  f"合约{contract}, 方向{'多头' if signal == 1 else '空头'}, "
                  f"价格{price:.0f}, 仓位{abs(self.position)}手")
            
            return True
        
        return False
    
    def close_position(self, price, date, reason):
        """平仓"""
        if self.position == 0:
            return
        
        if self.position > 0:
            price_pnl = (price - self.entry_price) * abs(self.position) * 10
        else:
            price_pnl = (self.entry_price - price) * abs(self.position) * 10
        
        close_cost = price * abs(self.position) * 10 * (self.transaction_cost + self.slippage)
        
        entry_cost = 0
        for trade in reversed(self.trades):
            if trade.get('entry_cost') is not None:
                entry_cost = trade['entry_cost']
                break
        total_cost = entry_cost + close_cost
        
        net_pnl = price_pnl - total_cost
        
        old_capital = self.capital
        self.capital += net_pnl
        
        margin_to_release = price * abs(self.position) * 10 * self.margin_rate
        self.margin_used -= margin_to_release
        
        print(f"📊 平仓 [{date.strftime('%Y-%m-%d')}]: "
              f"合约{self.position_contract}, 价格{price:.0f}, "
              f"盈亏{net_pnl:,.0f}元, 现金{old_capital:,.0f}→{self.capital:,.0f}元")
        
        self.trades.append({
            'date': date,
            'action': '平仓',
            'contract': self.position_contract,
            'direction': '多头' if self.position > 0 else '空头',
            'price': price,
            'position': self.position,
            'price_pnl': price_pnl,
            'close_cost': close_cost,
            'entry_cost': entry_cost,
            'total_cost': total_cost,
            'pnl': net_pnl,
            'reason': reason
        })
        
        self.position = 0
        self.position_contract = None
        self.entry_price = 0
        self.stop_loss_price = 0
        self.position_value = 0
    
    def check_stop_loss(self, current_price, date):
        """检查止损"""
        if self.position == 0:
            return False
        
        should_stop = False
        if self.position > 0 and current_price <= self.stop_loss_price:
            should_stop = True
        elif self.position < 0 and current_price >= self.stop_loss_price:
            should_stop = True
        
        if should_stop:
            self.close_position(current_price, date, f"止损触发 (止损价: {self.stop_loss_price:.0f})")
            return True
        
        return False
    
    def preprocess_data(self, df):
        """数据预处理 - 日线数据版本"""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 确保有必要的列
        if 'main_contract' not in df.columns:
            raise ValueError("数据文件缺少 main_contract 列，请使用新的数据获取工具")
        
        # 筛选主力合约数据
        main_data = df[df['contract'] == df['main_contract']].copy()
        
        if len(main_data) == 0:
            raise ValueError("没有找到主力合约数据")
        
        # 去重并按时间排序
        main_data = main_data.drop_duplicates(subset=['datetime']).sort_values('datetime')
        main_data = main_data.set_index('datetime')
        
        # 计算技术指标
        main_data['month'] = main_data.index.month
        main_data['rsi'] = self.calculate_rsi(main_data['close'], self.rsi_period)
        main_data['price_position'] = self.calculate_price_position(main_data['close'], self.price_pos_period)
        main_data['volatility'] = self.calculate_volatility(main_data['close'])
        main_data['vol_mean'] = main_data['volatility'].rolling(window=20).mean()
        main_data['vol_ratio'] = main_data['volatility'] / main_data['vol_mean']
        
        # 返回原始数据和处理后的主力合约日线数据
        return df, main_data.dropna()
    
    def backtest(self, df, daily_indicators):
        """执行回测 - 完美换仓版本"""
        results = []
        
        print(f"\n开始回测，共{len(daily_indicators)}个交易日")
        print(f"日期范围: {daily_indicators.index[0].date()} 到 {daily_indicators.index[-1].date()}")
        
        for i in range(len(daily_indicators)):
            current_date = daily_indicators.index[i]
            current_indicators = daily_indicators.iloc[i]
            main_contract = current_indicators['main_contract']
            
            # 不显示调试信息 - 减少日志噪音
            
            # 使用日线数据，直接从daily_indicators获取数据
            current_day_data = df[df['datetime'].dt.date == current_date.date()]
            
            if len(current_day_data) == 0:
                continue
            
            current_price = current_indicators['close']
            
            # 完美换仓逻辑
            if self.position != 0 and self.position_contract:
                # 检查是否应该开始换仓监控
                delivery_date = self.get_contract_delivery_date(self.position_contract)
                days_to_delivery = (delivery_date - current_date).days
                
                # 调试信息
                if i < 5:  # 前5天显示调试信息
                    print(f"  [{current_date.strftime('%Y-%m-%d')}] 持仓{self.position_contract}, "
                          f"距离交割{days_to_delivery}天, 是否监控: {days_to_delivery <= self.rollover_days_before_delivery}")
                
                # 首先检查是否已进入合约交割月份
                contract_year = int('20' + self.position_contract[2:4])
                contract_month = int(self.position_contract[4:6])
                current_year = current_date.year
                current_month = current_date.month
                
                # 检查是否需要在交割月前强制换仓
                force_rollover = False
                
                if current_year == contract_year and current_month == contract_month:
                    # 在交割月份，检查是否接近月底
                    from calendar import monthrange
                    last_day_of_month = monthrange(current_year, current_month)[1]
                    days_left_in_month = last_day_of_month - current_date.day
                    
                    if days_left_in_month <= 1:  # 交割月最后2天强制换仓
                        print(f"  🚨 [{current_date.strftime('%Y-%m-%d')}] {self.position_contract}交割月最后{days_left_in_month + 1}天，必须立即换仓！")
                        force_rollover = True
                        
                elif current_year == contract_year and current_month == contract_month - 1:
                    # 在交割月前一个月，检查是否是月底最后几天
                    from calendar import monthrange
                    last_day_of_month = monthrange(current_year, current_month)[1]
                    days_left_in_month = last_day_of_month - current_date.day
                    
                    if days_left_in_month <= 2:  # 交割月前最后3天强制换仓
                        print(f"  🚨 [{current_date.strftime('%Y-%m-%d')}] {self.position_contract}交割前最后{days_left_in_month + 1}天，必须立即换仓！")
                        force_rollover = True
                        
                elif current_year > contract_year or (current_year == contract_year and current_month > contract_month):
                    # 已经过了交割月份，绝对不能持有
                    print(f"  🚨 [{current_date.strftime('%Y-%m-%d')}] {self.position_contract}已过交割月份，绝对不能持有！")
                    force_rollover = True
                
                if force_rollover:
                    
                    # 寻找下一个合约
                    next_contract = self.get_next_contract(self.position_contract)
                    next_contract_data = current_day_data[current_day_data['contract'] == next_contract]
                    
                    if len(next_contract_data) > 0:
                        # 执行强制换仓（有成本）
                        current_contract_data = current_day_data[current_day_data['contract'] == self.position_contract]
                        if len(current_contract_data) > 0:
                            old_price = current_contract_data.iloc[-1]['close']
                            new_price = next_contract_data.iloc[-1]['close']
                            spread = (new_price - old_price) / old_price
                            cost = abs(self.position) * abs(spread * old_price) * self.contract_multiplier * (self.transaction_cost + self.slippage)
                            
                            print(f"⚡ [{current_date.strftime('%Y-%m-%d')}] 强制换仓: {self.position_contract}({old_price:.0f})→{next_contract}({new_price:.0f}) 价差{spread*100:+.2f}% 成本{cost:.0f}元 | 交割月到期")
                            
                            self.capital -= cost
                            self.rollover_costs += cost
                        
                        self.position_contract = next_contract
                        print(f"  强制切换持仓合约到: {next_contract}")
                        continue  # 换仓完成，跳过当天后续检查
                    else:
                        print(f"  ❌ 找不到可用合约，强制平仓")
                        # 强制平仓，使用当前合约的最后价格
                        if len(current_contract_data) > 0:
                            close_price = current_contract_data.iloc[-1]['close']
                            
                            # 通过正常交易流程执行强制平仓
                            if self.position != 0:
                                # 只平仓，不开新仓
                                self.close_position(close_price, current_date, "强制平仓：交割月到期")
                                print(f"  ✅ 强制平仓完成，当前持仓: {self.position}")
                        return
                
                # 然后检查当前持仓合约是否还有数据
                current_contract_data = current_day_data[current_day_data['contract'] == self.position_contract]
                
                if len(current_contract_data) == 0:
                    # 当前合约无数据，必须换仓到有数据的合约
                    print(f"  ⚠️ [{current_date.strftime('%Y-%m-%d')}] 持仓合约{self.position_contract}无数据，寻找可用合约")
                    
                    # 寻找有数据的下一个合约
                    available_contracts = current_day_data['contract'].unique()
                    next_contract = None
                    
                    # 按月份顺序寻找
                    current_month = int(self.position_contract[4:6])
                    current_year = int(self.position_contract[2:4])
                    
                    for i in range(1, 6):  # 最多找未来5个月
                        test_month = current_month + i
                        test_year = current_year
                        if test_month > 12:
                            test_month -= 12
                            test_year += 1
                        
                        test_contract = f"JD{test_year:02d}{test_month:02d}"
                        if test_contract in available_contracts:
                            next_contract = test_contract
                            break
                    
                    if next_contract:
                        print(f"  找到可用合约: {next_contract}，立即换仓")
                        # 强制换仓，不需要价差计算
                        self.position_contract = next_contract
                        print(f"  强制切换持仓合约到: {next_contract}")
                    else:
                        print(f"  ❌ 找不到可用合约，无法继续交易")
                
                elif self.should_start_rollover_monitoring(current_date, self.position_contract, df):
                    next_contract = self.get_next_contract(self.position_contract)
                    
                    # 检查下一个合约是否有数据
                    next_contract_data = current_day_data[current_day_data['contract'] == next_contract]
                    if len(next_contract_data) == 0:
                        if i < 10:  # 只在前10天显示这个信息
                            print(f"  ⚠️ [{current_date.strftime('%Y-%m-%d')}] 下一个合约{next_contract}无数据，等待上市")
                    else:
                        # 计算价差
                        spread_data = self.calculate_spread(df, current_date, self.position_contract, next_contract)
                        
                        if spread_data:
                            # 记录监控数据
                            monitor_key = f"{self.position_contract}_{next_contract}"
                            if monitor_key not in self.spread_monitoring:
                                self.spread_monitoring[monitor_key] = []
                            self.spread_monitoring[monitor_key].append(spread_data)
                            
                            # 简化监控日志 - 不打印每日监控细节
                            pass
                            
                        # 判断是否执行换仓
                        should_rollover, reason = self.should_execute_rollover(
                            spread_data, self.spread_monitoring[monitor_key])
                        
                        if should_rollover:
                            self.execute_rollover(spread_data, reason)
                        else:
                            # 不显示等待原因 - 减少日志噪音
                            pass
            
            # 使用持仓合约的价格
            if self.position != 0 and self.position_contract:
                contract_data = current_day_data[current_day_data['contract'] == self.position_contract]
                if len(contract_data) > 0:
                    current_price = contract_data.iloc[-1]['close']
            
            # 检查止损
            self.check_stop_loss(current_price, current_date)
            
            # 交易信号
            month = current_indicators['month']
            price_pos = current_indicators['price_position']
            rsi = current_indicators['rsi']
            vol_ratio = current_indicators['vol_ratio']
            
            if vol_ratio > self.vol_max:
                continue
            
            signal = 0
            reason = ''
            
            # 买入信号
            if month in self.buy_months and self.position <= 0:
                # 不显示调试信息 - 减少日志噪音
                
                if price_pos <= 0.7 and rsi <= 80:
                    signal = 1
                    reason = f'买入信号: 月份{month}, 价格位置{price_pos:.2f}, RSI{rsi:.1f}'
                    # 使用当天的主力合约开仓
                    trade_contract = main_contract
                    # 简化日志 - 开仓时会显示详细信息
            
            # 卖出信号
            elif month in self.sell_months and self.position >= 0:
                if price_pos >= self.sell_threshold and rsi >= self.rsi_sell_min:
                    signal = -1
                    reason = f'卖出信号: 月份{month}, 价格位置{price_pos:.2f}, RSI{rsi:.1f}'
                    # 使用当天的主力合约开仓
                    trade_contract = main_contract
                    # 简化日志 - 开仓时会显示详细信息
            
            # 执行交易
            if signal != 0:
                # 检查选择的合约是否即将到期（不能开仓当月或下月合约）
                contract_year = int('20' + trade_contract[2:4])
                contract_month = int(trade_contract[4:6])
                current_year = current_date.year
                current_month = current_date.month
                
                # 只在交割月最后几天才禁止开仓
                should_avoid_contract = False
                
                if contract_year == current_year and contract_month == current_month:
                    # 在交割月份，检查是否接近月底
                    from calendar import monthrange
                    last_day_of_month = monthrange(current_year, current_month)[1]
                    days_left_in_month = last_day_of_month - current_date.day
                    
                    if days_left_in_month <= 5:  # 交割月最后5天才禁止开仓
                        should_avoid_contract = True
                        print(f"  ⚠️ [{current_date.strftime('%Y-%m-%d')}] 合约{trade_contract}交割月最后{days_left_in_month + 1}天，寻找远月合约")
                
                if should_avoid_contract:
                    
                    # 寻找至少2个月后的合约
                    target_month = current_month + 2
                    target_year = current_year
                    if target_month > 12:
                        target_month -= 12
                        target_year += 1
                    
                    # 尝试找到可用的远月合约
                    for i in range(6):  # 最多找6个月后的合约
                        test_month = target_month + i
                        test_year = target_year
                        if test_month > 12:
                            test_month -= 12
                            test_year += 1
                        
                        test_contract = f"JD{test_year % 100:02d}{test_month:02d}"
                        test_contract_data = current_day_data[current_day_data['contract'] == test_contract]
                        
                        if len(test_contract_data) > 0:
                            trade_contract = test_contract
                            print(f"  🎯 改选远月合约{trade_contract}开仓")
                            break
                    else:
                        print(f"  ❌ 找不到合适的远月合约，跳过开仓")
                        continue
                
                # 验证选择的合约确实是主力合约（如果没有改选的话）
                day_volumes = current_day_data.groupby('contract')['volume'].sum().sort_values(ascending=False)
                actual_main = day_volumes.index[0] if len(day_volumes) > 0 else None
                
                trade_contract_data = current_day_data[current_day_data['contract'] == trade_contract]
                if len(trade_contract_data) > 0:
                    trade_price = trade_contract_data.iloc[-1]['close']
                    self.execute_trade(signal, trade_price, trade_contract, reason, current_date)
                else:
                    print(f"  ❌ 合约{trade_contract}无数据，无法开仓")
            
            results.append({
                'date': current_date,
                'main_contract': main_contract,
                'position_contract': self.position_contract,
                'position': self.position,
                'capital': self.capital,
                'price': current_price
            })
        
        # 最后平仓
        if self.position != 0:
            last_date = daily_indicators.index[-1]
            last_data = df[df['datetime'].dt.date == last_date.date()]
            if self.position_contract:
                contract_data = last_data[last_data['contract'] == self.position_contract]
                if len(contract_data) > 0:
                    last_price = contract_data.iloc[-1]['close']
                    self.close_position(last_price, last_date, "回测结束")
        
        return pd.DataFrame(results)
    
    def calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.trades:
            return {}
        
        # 计算每日收益率
        daily_returns = []
        for trade in self.trades:
            if 'pnl' in trade and trade['pnl'] != 0:
                daily_return = trade['pnl'] / self.initial_capital
                daily_returns.append(daily_return)
        
        if not daily_returns:
            return {}
        
        # 计算指标
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        sharpe_ratio = (total_return * 100) / (volatility * 100) if volatility > 0 else 0
        
        # 最大回撤（简化计算）
        cumulative_returns = np.cumsum(daily_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / (1 + peak)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_return': total_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100
        }
    
    def print_summary(self):
        """打印简洁的策略总结"""
        print("\n" + "="*80)
        print("📊 策略执行总结")
        print("="*80)
        
        print(f"\n💰 资金情况:")
        print(f"  初始资金: {self.initial_capital:,} 元")
        print(f"  最终资金: {self.capital:,} 元")
        print(f"  总收益: {self.capital - self.initial_capital:,} 元")
        print(f"  收益率: {(self.capital/self.initial_capital - 1)*100:.1f}%")
        
        print(f"\n📋 交易统计:")
        open_trades = [t for t in self.trades if t['action'] == '开仓']
        close_trades = [t for t in self.trades if t['action'] == '平仓']
        print(f"  开仓次数: {len(open_trades)}")
        print(f"  胜率: {len([t for t in close_trades if t['pnl'] > 0])/len(close_trades)*100:.1f}%" if close_trades else "  胜率: N/A")
        print(f"  换仓次数: {len(self.rollover_records)}")
        print(f"  换仓总成本: {self.rollover_costs:,} 元")
        
        # 性能指标
        metrics = self.calculate_performance_metrics()
        if metrics:
            print(f"\n📈 风险指标:")
            print(f"  年化波动率: {metrics['volatility']:.1f}%")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {metrics['max_drawdown']:.1f}%")
        
        print(f"\n✅ 策略执行完成！")
    
    def run_strategy(self, data_file):
        """运行策略"""
        print("🚀 启动鸡蛋期货策略（完美换仓版）...")
        
        print(f"\n📊 加载数据: {data_file}")
        df = pd.read_csv(data_file)
        print(f"数据条数: {len(df)} 条")
        
        print("\n🔧 数据预处理...")
        df, daily_indicators = self.preprocess_data(df)
        print(f"处理后日线数据: {len(daily_indicators)} 条")
        
        print(f"\n📈 执行策略回测（初始资金: {self.initial_capital:,}元）...")
        results = self.backtest(df, daily_indicators)
        
        self.print_summary()
        
        return results

def main():
    """主函数"""
    config = {
        'initial_capital': 20000,
        'data_file': 'jd_all_contracts_daily_2022-2025_20250911_145111.csv'  # 使用新的日线数据
    }
    
    strategy = JDStrategyPerfectRollover(initial_capital=config['initial_capital'])
    
    if not os.path.exists(config['data_file']):
        print(f"❌ 数据文件不存在: {config['data_file']}")
        return None
    
    results = strategy.run_strategy(config['data_file'])
    
    return strategy, results

if __name__ == "__main__":
    strategy, results = main()
    if strategy:
        pass  # 完成日志已在print_summary中显示
