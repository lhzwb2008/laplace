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
        
        # 换仓参数
        self.rollover_days_before_delivery = 30  # 交割前30天开始监控
        self.max_acceptable_spread = 0.02  # 最大可接受价差2%
        self.target_spread = 0.005  # 目标价差0.5%
        
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
        """获取合约的交割日期（每月15日）"""
        year = int('20' + contract_code[2:4])
        month = int(contract_code[4:6])
        delivery_date = datetime(year, month, 15)
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
    
    def should_start_rollover_monitoring(self, current_date, contract):
        """判断是否应该开始换仓监控（基于固定的交割前30天）"""
        if not contract:
            return False
        
        delivery_date = self.get_contract_delivery_date(contract)
        days_to_delivery = (delivery_date - current_date).days
        
        return days_to_delivery <= self.rollover_days_before_delivery
    
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
        """判断是否应该执行换仓"""
        if not spread_data:
            return False
        
        current_spread = spread_data['spread']
        abs_spread = abs(current_spread)
        
        # 条件1：价差小于目标价差（0.5%），立即换仓
        if abs_spread <= self.target_spread:
            return True, f"价差{current_spread*100:.2f}%接近零"
        
        # 条件2：价差小于最大可接受价差（2%），且是近期最优时机
        if abs_spread <= self.max_acceptable_spread:
            if len(monitoring_history) >= 3:
                recent_spreads = [h['abs_spread'] for h in monitoring_history[-7:]]  # 最近7天
                if abs_spread <= min(recent_spreads):
                    return True, f"价差{current_spread*100:.2f}%为近期最优"
        
        # 条件3：新合约成交量明显更大（主力已切换）
        total_volume = spread_data['old_volume'] + spread_data['new_volume']
        if total_volume > 0:
            new_volume_ratio = spread_data['new_volume'] / total_volume
            if new_volume_ratio > 0.8 and abs_spread <= self.max_acceptable_spread:
                return True, f"新合约成交量占比{new_volume_ratio*100:.1f}%，主力已切换"
        
        # 条件4：强制换仓（距离交割日7天内）
        delivery_date = self.get_contract_delivery_date(spread_data['old_contract'])
        days_to_delivery = (delivery_date - spread_data['date']).days
        if days_to_delivery <= 7:
            return True, f"距离交割仅{days_to_delivery}天，强制换仓"
        
        return False, ""
    
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
        
        print(f"📋 换仓 [{spread_data['date'].strftime('%Y-%m-%d')}]: "
              f"{old_contract}({old_price:.0f}) → {new_contract}({new_price:.0f}), "
              f"价差{spread_data['spread']*100:+.2f}%, 成本{rollover_cost:.0f}元")
        print(f"    理由: {reason}")
    
    def calculate_rollover_cost(self, old_price, new_price, position_size):
        """计算换仓成本"""
        # 价差成本（如果新合约更贵）
        price_diff_cost = 0
        if new_price > old_price:
            price_diff_cost = (new_price - old_price) * abs(position_size) * self.contract_multiplier
        
        # 交易成本（双边）
        transaction_cost = (old_price + new_price) * abs(position_size) * \
                          self.contract_multiplier * (self.transaction_cost + self.slippage)
        
        return price_diff_cost + transaction_cost
    
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
        """数据预处理"""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        if 'main_contract' not in df.columns:
            df['date'] = df['datetime'].dt.date
            daily_volume = df.groupby(['date', 'contract'])['volume'].sum().reset_index()
            main_contracts = daily_volume.loc[daily_volume.groupby('date')['volume'].idxmax()]
            main_contracts = main_contracts.rename(columns={'contract': 'main_contract'})
            df = df.merge(main_contracts[['date', 'main_contract']], on='date', how='left')
        else:
            df['date'] = df['datetime'].dt.date
        
        main_data = df[df['contract'] == df['main_contract']].copy()
        
        daily_main = main_data.set_index('datetime').resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'hold': 'last',
            'main_contract': 'last'
        }).dropna()
        
        daily_main['month'] = daily_main.index.month
        daily_main['rsi'] = self.calculate_rsi(daily_main['close'], self.rsi_period)
        daily_main['price_position'] = self.calculate_price_position(daily_main['close'], self.price_pos_period)
        daily_main['volatility'] = self.calculate_volatility(daily_main['close'])
        daily_main['vol_mean'] = daily_main['volatility'].rolling(window=20).mean()
        daily_main['vol_ratio'] = daily_main['volatility'] / daily_main['vol_mean']
        
        return df, daily_main.dropna()
    
    def backtest(self, df, daily_indicators):
        """执行回测 - 完美换仓版本"""
        results = []
        
        print(f"\n开始回测，共{len(daily_indicators)}个交易日")
        
        for i in range(len(daily_indicators)):
            current_date = daily_indicators.index[i]
            current_indicators = daily_indicators.iloc[i]
            main_contract = current_indicators['main_contract']
            
            current_day_data = df[df['datetime'].dt.date == current_date.date()]
            main_contract_data = current_day_data[current_day_data['contract'] == main_contract]
            
            if len(main_contract_data) == 0:
                continue
            
            current_price = main_contract_data.iloc[-1]['close']
            
            # 完美换仓逻辑
            if self.position != 0 and self.position_contract:
                # 检查是否应该开始换仓监控
                delivery_date = self.get_contract_delivery_date(self.position_contract)
                days_to_delivery = (delivery_date - current_date).days
                
                # 调试信息
                if i < 5:  # 前5天显示调试信息
                    print(f"  [{current_date.strftime('%Y-%m-%d')}] 持仓{self.position_contract}, "
                          f"距离交割{days_to_delivery}天, 是否监控: {days_to_delivery <= self.rollover_days_before_delivery}")
                
                # 首先检查当前持仓合约是否还有数据
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
                
                elif self.should_start_rollover_monitoring(current_date, self.position_contract):
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
                            
                            print(f"  📊 [{current_date.strftime('%Y-%m-%d')}] 监控换仓: "
                                  f"{self.position_contract}({spread_data['old_price']:.0f}) vs "
                                  f"{next_contract}({spread_data['new_price']:.0f}), "
                                  f"价差{spread_data['spread']*100:+.2f}%")
                            
                            # 判断是否执行换仓
                            should_rollover, reason = self.should_execute_rollover(
                                spread_data, self.spread_monitoring[monitor_key])
                            
                            if should_rollover:
                                self.execute_rollover(spread_data, reason)
            
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
                if price_pos <= 0.7 and rsi <= 80:
                    signal = 1
                    reason = f'买入信号: 月份{month}, 价格位置{price_pos:.2f}, RSI{rsi:.1f}'
                    trade_contract = main_contract
            
            # 卖出信号
            elif month in self.sell_months and self.position >= 0:
                if price_pos >= self.sell_threshold and rsi >= self.rsi_sell_min:
                    signal = -1
                    reason = f'卖出信号: 月份{month}, 价格位置{price_pos:.2f}, RSI{rsi:.1f}'
                    trade_contract = main_contract
            
            # 执行交易
            if signal != 0:
                trade_contract_data = current_day_data[current_day_data['contract'] == trade_contract]
                if len(trade_contract_data) > 0:
                    trade_price = trade_contract_data.iloc[-1]['close']
                    self.execute_trade(signal, trade_price, trade_contract, reason, current_date)
            
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
    
    def print_summary(self):
        """打印策略总结"""
        print("\n" + "="*80)
        print("📊 完美换仓策略执行总结")
        print("="*80)
        
        print(f"\n💰 资金情况:")
        print(f"  初始资金: {self.initial_capital:,.0f} 元")
        print(f"  最终资金: {self.capital:,.0f} 元")
        print(f"  总收益: {self.capital - self.initial_capital:,.0f} 元")
        print(f"  收益率: {(self.capital/self.initial_capital - 1)*100:.1f}%")
        
        print(f"\n📋 交易统计:")
        open_trades = [t for t in self.trades if t['action'] == '开仓']
        close_trades = [t for t in self.trades if t['action'] == '平仓']
        print(f"  开仓次数: {len(open_trades)}")
        print(f"  平仓次数: {len(close_trades)}")
        
        if close_trades:
            wins = [t for t in close_trades if t['pnl'] > 0]
            print(f"  胜率: {len(wins)/len(close_trades)*100:.1f}%")
        
        print(f"\n🔄 换仓统计:")
        print(f"  换仓次数: {len(self.rollover_records)}")
        print(f"  换仓总成本: {self.rollover_costs:,.0f} 元")
        
        if self.rollover_records:
            print(f"\n  详细换仓记录:")
            for i, record in enumerate(self.rollover_records):
                print(f"    {i+1}. {record['date'].strftime('%Y-%m-%d')}: "
                      f"{record['old_contract']} → {record['new_contract']}, "
                      f"价差{record['spread']:+.2f}%, 成本{record['cost']:.0f}元")
                print(f"       理由: {record['reason']}")
        
        # 打印价差监控统计
        if self.spread_monitoring:
            print(f"\n📈 价差监控统计:")
            for key, data in self.spread_monitoring.items():
                old_contract, new_contract = key.split('_')
                spreads = [d['spread'] for d in data]
                abs_spreads = [abs(s) for s in spreads]
                
                print(f"\n  {old_contract} → {new_contract}:")
                print(f"    监控天数: {len(data)} 天")
                print(f"    平均价差: {np.mean(spreads)*100:+.2f}%")
                print(f"    最小绝对价差: {min(abs_spreads)*100:.2f}%")
                print(f"    最大绝对价差: {max(abs_spreads)*100:.2f}%")
                
                # 显示最后几天的价差
                print(f"    最后3天价差: ", end="")
                for d in data[-3:]:
                    print(f"{d['spread']*100:+.2f}%({d['date'].strftime('%m-%d')}) ", end="")
                print()
    
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
        'data_file': 'jd_all_contracts_1min_20250910_104224.csv'
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
        print("\n✅ 完美换仓策略执行完成！")
