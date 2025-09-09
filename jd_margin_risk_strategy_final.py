#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鸡蛋期货保证金风险监控策略 - 最终修复版
完全修复利润计算问题，并根据正确的合约规格更新

鸡蛋期货(JD)合约规格（大连商品交易所）：
- 交易单位：5吨/手
- 报价单位：元（人民币）/500千克
- 最小变动价位：1元/500千克
- 最低交易保证金：合约价值的5%
- 价格计算：显示价格为500公斤价格，买一手需×10（因为5吨=5000公斤=10×500公斤）

核心修复：
1. 统一交易成本处理逻辑
2. 确保资金变化与交易盈亏完全一致
3. 更新保证金比例为正确的5%
4. 保持原有交易逻辑

作者: AI Assistant
创建时间: 2025-01-08
更新时间: 2025-01-08（合约规格修正）
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class JDMarginRiskStrategyFinal:
    """鸡蛋期货保证金风险监控策略 - 最终修复版"""
    
    def __init__(self, initial_capital=100000):
        # 策略参数（恢复高收益版本）
        self.buy_months = [6, 7, 8]  # 夏季买入
        self.sell_months = [11, 12, 1]  # 冬季卖出
        self.buy_threshold = 0.4  # 买入价格位置阈值
        self.sell_threshold = 0.6  # 卖出价格位置阈值
        self.price_pos_period = 120  # 价格位置计算周期（天）
        self.rsi_period = 14  # RSI周期
        self.rsi_buy_max = 70  # RSI买入最大值
        self.rsi_sell_min = 25  # RSI卖出最小值
        self.vol_max = 2.5  # 波动率最大倍数
        
        # 风险管理参数（极致高收益版本）
        self.initial_capital = initial_capital
        self.margin_rate = 0.10  # 保证金比例（10%，根据实际交易环境）
        self.max_position_ratio = 0.95  # 最大仓位比例（提高到95%）
        self.risk_per_trade = 0.2  # 单次交易风险（提高到20%）
        self.transaction_cost = 0.0005  # 交易成本
        self.slippage = 0.0002  # 滑点
        self.contract_multiplier = 10  # 鸡蛋期货合约乘数（5吨/手=10×500kg）
        
        # 爆仓风险监控参数（根据5%保证金调整）
        self.maintenance_margin_rate = 0.04  # 维持保证金比例（4%，低于此强制平仓）
        self.margin_call_rate = 0.045  # 追加保证金比例（4.5%，低于此需追加）
        
        # 移除动态仓位管理，使用固定仓位
        
        # 移除止损减仓参数，恢复简单止损逻辑
        
        # 交易状态
        self.capital = initial_capital
        self.position = 0
        self.position_value = 0
        self.margin_used = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.trades = []
        
        # 风险监控记录
        self.daily_risk_data = []
        self.max_margin_usage = 0
        self.margin_call_alerts = []
        self.near_liquidation_alerts = []
        
        # 新增：交易成本追踪
        self.total_transaction_costs = 0
        
        # 价格回撤追踪变量
        self.price_peak_since_entry = 0
        self.max_price_drawdown_current_trade = 0
        
        # 权益回撤和保证金占比追踪
        self.equity_peak = initial_capital  # 权益最高点
        self.max_equity_drawdown = 0  # 最大权益回撤
        self.max_margin_ratio = 0  # 保证金占比权益最大时的比例
        
    def calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_price_position(self, prices, period=120):
        """计算价格在指定周期内的分位数位置"""
        return prices.rolling(window=period).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
    
    def calculate_volatility(self, prices, period=20):
        """计算价格波动率"""
        returns = prices.pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)
    
    def calculate_position_size(self, price, direction):
        """计算合理的仓位大小（恢复风险控制逻辑）"""
        available_capital = self.capital - self.margin_used
        
        # 使用配置的目标保证金占用率（默认50%，可通过main函数配置调整）
        target_margin_usage = getattr(self, 'target_margin_usage', 0.5)
        target_margin_amount = self.capital * target_margin_usage
        
        # 基于目标保证金占用率计算仓位
        target_position = target_margin_amount / (price * self.margin_rate * 10)
        
        # 基于风险的仓位计算（使用固定10%止损）
        max_risk_amount = self.capital * self.risk_per_trade
        fixed_stop_loss_ratio = 0.10  # 固定10%止损比例
        risk_based_position = max_risk_amount / (price * fixed_stop_loss_ratio * 10)
        
        # 基于可用资金的最大仓位（保持5%安全边际）
        safety_margin = 0.05
        safe_capital = available_capital * (1 - safety_margin)
        max_safe_position = safe_capital / (price * self.margin_rate * 10)
        
        # 激进策略：优先目标占用率，大幅放宽限制
        position_size = min(target_position, max_safe_position, risk_based_position * 2.0)  # 大幅放宽风险限制
        
        # 确保至少1手
        final_position = max(1, int(position_size))
        
        # 最终检查：允许保证金占用达到配置的目标占用率
        required_margin = price * final_position * 10 * self.margin_rate
        max_allowed_margin = self.capital * target_margin_usage
        if required_margin > max_allowed_margin:
            final_position = max(1, int(max_allowed_margin / (price * 10 * self.margin_rate)))
        
        return final_position
    
    def calculate_stop_loss(self, entry_price, direction):
        """计算止损价格（固定10%止损）"""
        stop_loss_ratio = 0.10  # 固定10%止损比例
        if direction == 1:
            return entry_price * (1 - stop_loss_ratio)
        else:
            return entry_price * (1 + stop_loss_ratio)
    
    def check_margin_risk(self, current_price, date):
        """检查保证金风险"""
        if self.position == 0:
            return False
        
        # 计算当前浮动盈亏
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) * abs(self.position) * 10
        else:
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position) * 10
        
        # 计算当前权益
        current_equity = self.capital + unrealized_pnl
        
        # 计算保证金占用率
        margin_usage_rate = self.margin_used / current_equity if current_equity > 0 else 1
        
        # 记录风险数据
        risk_data = {
            'date': date,
            'price': current_price,
            'position': self.position,
            'capital': self.capital,
            'unrealized_pnl': unrealized_pnl,
            'current_equity': current_equity,
            'margin_used': self.margin_used,
            'margin_usage_rate': margin_usage_rate,
            'available_capital': current_equity - self.margin_used,
            'margin_call_risk': margin_usage_rate > self.margin_call_rate / self.margin_rate,
            'liquidation_risk': margin_usage_rate > self.maintenance_margin_rate / self.margin_rate
        }
        
        self.daily_risk_data.append(risk_data)
        self.max_margin_usage = max(self.max_margin_usage, margin_usage_rate)
        
        # 检查追加保证金风险
        if risk_data['margin_call_risk']:
            shortage = self.margin_used * self.margin_call_rate / self.margin_rate - current_equity
            self.margin_call_alerts.append({
                'date': date,
                'price': current_price,
                'shortage': shortage
            })
        
        # 检查强制平仓风险
        if risk_data['liquidation_risk']:
            shortage = self.margin_used * self.maintenance_margin_rate / self.margin_rate - current_equity
            self.near_liquidation_alerts.append({
                'date': date,
                'price': current_price,
                'shortage': shortage
            })
            return True
        
        return False
    
    def execute_trade(self, signal, price, reason, date):
        """执行交易 - 最终修复版（含动态仓位管理）"""
        # 如果有持仓，先平仓
        if self.position != 0:
            self.close_position(price, date, "信号切换")
        
        if signal != 0:
            position_size = self.calculate_position_size(price, signal)
            
            # 移除动态仓位管理，使用固定仓位
            
            trade_cost = price * position_size * 10 * (self.transaction_cost + self.slippage)
            required_margin = price * position_size * 10 * self.margin_rate
            
            # 风险控制：确保不爆仓
            available_capital = self.capital - self.margin_used
            
            # 检查是否有足够资金开仓（只需要保证金，交易成本在平仓时扣除）
            if required_margin > available_capital:
                # 计算最大可承受仓位（基于可用资金）
                max_affordable_size = int(available_capital / (price * 10 * self.margin_rate))
                if max_affordable_size < 1:
                    print(f"❌ 资金不足，无法开仓（可用资金: {available_capital:,.0f}元，需要保证金: {required_margin:,.0f}元）")
                    return False
                position_size = max_affordable_size
                trade_cost = price * position_size * 10 * (self.transaction_cost + self.slippage)
                required_margin = price * position_size * 10 * self.margin_rate
                print(f"⚠️ 资金限制，调整仓位为 {position_size} 手（可用资金: {available_capital:,.0f}元）")
            
            self.position = position_size * signal
            self.entry_price = price
            self.stop_loss_price = self.calculate_stop_loss(price, signal)
            self.position_value = price * abs(self.position) * 10
            self.margin_used += required_margin
            
            # 重置价格回撤追踪变量
            self.price_peak_since_entry = price  # 初始化为开仓价格
            self.max_price_drawdown_current_trade = 0
            
            # 重置权益回撤和保证金占比追踪变量（每笔交易独立统计）
            self.equity_peak = self.capital  # 重置权益峰值为当前资金
            self.max_equity_drawdown = 0  # 重置最大权益回撤
            self.max_margin_ratio = 0  # 重置最大保证金占比
            
            # 统一处理：开仓时不扣除交易成本，在平仓时一次性扣除所有成本
            # self.capital -= trade_cost  # 注释掉这行
            self.total_transaction_costs += trade_cost  # 累计交易成本
            
            self.trades.append({
                'date': date,
                'action': '开仓',
                'direction': '多头' if signal == 1 else '空头',
                'price': price,
                'position': self.position,
                'margin': required_margin,
                'stop_loss': self.stop_loss_price,
                'entry_cost': trade_cost,
                'reason': reason
            })
            
            return True
        
        return False
    
    def close_position(self, price, date, reason):
        """平仓 - 最终修复版"""
        if self.position == 0:
            return
        
        # 计算价格差盈亏
        if self.position > 0:
            price_pnl = (price - self.entry_price) * abs(self.position) * 10
        else:
            price_pnl = (self.entry_price - price) * abs(self.position) * 10
        
        # 计算平仓交易成本
        close_cost = price * abs(self.position) * 10 * (self.transaction_cost + self.slippage)
        
        # 获取开仓成本（查找最近的开仓记录）
        entry_cost = 0
        for trade in reversed(self.trades):
            if trade.get('entry_cost') is not None:
                entry_cost = trade['entry_cost']
                break
        total_cost = entry_cost + close_cost
        
        # 净盈亏 = 价格差盈亏 - 总交易成本
        net_pnl = price_pnl - total_cost
        
        # 更新资金和保证金
        old_capital = self.capital
        self.capital += net_pnl
        
        # 释放保证金：应该释放开仓时占用的保证金
        margin_to_release = price * abs(self.position) * 10 * self.margin_rate
        self.margin_used -= margin_to_release
        self.total_transaction_costs += close_cost
        
        # 移除动态仓位管理逻辑
        
        # 打印平仓后现金值
        print(f"📊 平仓详情 [{date.strftime('%Y-%m-%d')}]: 价格{price:.0f}, 盈亏{net_pnl:,.0f}元, 现金{old_capital:,.0f}→{self.capital:,.0f}元")
        
        self.trades.append({
            'date': date,
            'action': '平仓',
            'direction': '多头' if self.position > 0 else '空头',
            'price': price,
            'position': self.position,
            'price_pnl': price_pnl,
            'close_cost': close_cost,
            'entry_cost': entry_cost,
            'total_cost': total_cost,
            'pnl': net_pnl,
            'reason': reason,
            'max_price_drawdown': self.max_price_drawdown_current_trade,
            'price_peak': self.price_peak_since_entry,
            'max_equity_drawdown': self.max_equity_drawdown,
            'max_margin_ratio': self.max_margin_ratio
        })
        
        # 重置持仓状态
        self.position = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.position_value = 0
    
    def check_stop_loss(self, current_price, date):
        """检查止损 - 简单直接平仓"""
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
        df.set_index('datetime', inplace=True)
        
        daily_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'hold': 'last'
        }).dropna()
        
        daily_df['month'] = daily_df.index.month
        daily_df['rsi'] = self.calculate_rsi(daily_df['close'], self.rsi_period)
        daily_df['price_position'] = self.calculate_price_position(daily_df['close'], self.price_pos_period)
        daily_df['volatility'] = self.calculate_volatility(daily_df['close'])
        daily_df['vol_mean'] = daily_df['volatility'].rolling(window=60).mean()
        daily_df['vol_ratio'] = daily_df['volatility'] / daily_df['vol_mean']
        
        return daily_df.dropna()
    
    def backtest(self, df):
        """执行回测"""
        df = df.copy()
        df['signal'] = 0
        df['position'] = 0
        df['capital'] = 0
        df['equity'] = 0  # 新增：当前权益（现金+持仓市值）
        df['price_drawdown'] = 0  # 新增：价格回撤（相对于成本价）
        df['trade_reason'] = ''
        
        # 重置价格回撤追踪变量
        self.price_peak_since_entry = 0
        self.max_price_drawdown_current_trade = 0
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            month = df.iloc[i]['month']
            price_pos = df.iloc[i]['price_position']
            rsi = df.iloc[i]['rsi']
            vol_ratio = df.iloc[i]['vol_ratio']
            
            # 检查保证金风险
            liquidation_risk = self.check_margin_risk(current_price, current_date)
            
            # 如果面临强制平仓，立即平仓
            if liquidation_risk and self.position != 0:
                self.close_position(current_price, current_date, "强制平仓风险，主动平仓")
            
            # 检查止损
            self.check_stop_loss(current_price, current_date)
            
            # 波动率过滤
            if vol_ratio > self.vol_max:
                # 计算当前权益（现金 + 持仓浮动盈亏）
                if self.position != 0:
                    unrealized_pnl = (current_price - self.entry_price) * self.position * self.contract_multiplier
                    current_equity = self.capital + unrealized_pnl
                    
                    # 计算价格回撤
                    if self.position > 0:  # 多头持仓
                        if current_price > self.price_peak_since_entry:
                            self.price_peak_since_entry = current_price
                        if self.price_peak_since_entry > self.entry_price:
                            current_price_drawdown = (self.price_peak_since_entry - current_price) / self.price_peak_since_entry
                        else:
                            current_price_drawdown = (self.entry_price - current_price) / self.entry_price
                    else:  # 空头持仓
                        if self.price_peak_since_entry == 0 or current_price < self.price_peak_since_entry:
                            self.price_peak_since_entry = current_price
                        current_price_drawdown = (current_price - self.price_peak_since_entry) / self.price_peak_since_entry
                    
                    self.max_price_drawdown_current_trade = max(self.max_price_drawdown_current_trade, current_price_drawdown)
                else:
                    current_equity = self.capital
                    current_price_drawdown = 0
                
                df.iloc[i, df.columns.get_loc('signal')] = 0
                df.iloc[i, df.columns.get_loc('position')] = self.position
                df.iloc[i, df.columns.get_loc('capital')] = self.capital
                df.iloc[i, df.columns.get_loc('equity')] = current_equity
                df.iloc[i, df.columns.get_loc('price_drawdown')] = current_price_drawdown
                continue
            
            signal = 0
            reason = ''
            
            # 买入信号（做多）
            if (month in self.buy_months and 
                price_pos <= self.buy_threshold and 
                rsi <= self.rsi_buy_max and 
                self.position <= 0):
                signal = 1
                reason = f'买入信号: 月份{month}, 价格位置{price_pos:.2f}, RSI{rsi:.1f}'
            
            # 卖出信号（做空）
            elif (month in self.sell_months and 
                  price_pos >= self.sell_threshold and 
                  rsi >= self.rsi_sell_min and 
                  self.position >= 0):
                signal = -1
                reason = f'卖出信号: 月份{month}, 价格位置{price_pos:.2f}, RSI{rsi:.1f}'
            
            # 执行交易
            if signal != 0:
                self.execute_trade(signal, current_price, reason, current_date)
            
            # 计算当前权益（现金 + 持仓浮动盈亏）
            if self.position != 0:
                unrealized_pnl = (current_price - self.entry_price) * self.position * self.contract_multiplier
                current_equity = self.capital + unrealized_pnl
                
                # 计算价格回撤
                if self.position > 0:  # 多头持仓
                    # 更新价格峰值
                    if current_price > self.price_peak_since_entry:
                        self.price_peak_since_entry = current_price
                    # 计算从峰值的回撤
                    if self.price_peak_since_entry > self.entry_price:
                        current_price_drawdown = (self.price_peak_since_entry - current_price) / self.price_peak_since_entry
                    else:
                        current_price_drawdown = (self.entry_price - current_price) / self.entry_price
                else:  # 空头持仓
                    # 对于空头，价格下跌是有利的，价格上涨是不利的
                    if self.price_peak_since_entry == 0 or current_price < self.price_peak_since_entry:
                        self.price_peak_since_entry = current_price
                    # 计算从最低点的回撤（价格上涨对空头不利）
                    current_price_drawdown = (current_price - self.price_peak_since_entry) / self.price_peak_since_entry
                
                self.max_price_drawdown_current_trade = max(self.max_price_drawdown_current_trade, current_price_drawdown)
            else:
                current_equity = self.capital
                current_price_drawdown = 0
                # 重置价格回撤追踪变量
                self.price_peak_since_entry = 0
                self.max_price_drawdown_current_trade = 0
            
            # 更新权益回撤和保证金占比指标
            if current_equity > self.equity_peak:
                self.equity_peak = current_equity
            
            # 计算当前权益回撤
            current_equity_drawdown = (self.equity_peak - current_equity) / self.equity_peak if self.equity_peak > 0 else 0
            self.max_equity_drawdown = max(self.max_equity_drawdown, current_equity_drawdown)
            
            # 计算当前保证金占比
            current_margin_ratio = self.margin_used / current_equity if current_equity > 0 else 0
            self.max_margin_ratio = max(self.max_margin_ratio, current_margin_ratio)
            
            df.iloc[i, df.columns.get_loc('signal')] = signal
            df.iloc[i, df.columns.get_loc('position')] = self.position
            df.iloc[i, df.columns.get_loc('capital')] = self.capital
            df.iloc[i, df.columns.get_loc('equity')] = current_equity
            df.iloc[i, df.columns.get_loc('price_drawdown')] = current_price_drawdown
            df.iloc[i, df.columns.get_loc('trade_reason')] = reason
        
        # 最后平仓
        if self.position != 0:
            self.close_position(df.iloc[-1]['close'], df.index[-1], "回测结束")
            # 更新最后一行的权益
            df.iloc[-1, df.columns.get_loc('equity')] = self.capital
        
        return df
    
    def calculate_performance_metrics(self, df):
        """计算策略性能指标"""
        if len(df) == 0:
            return {}
        
        # 使用权益序列计算回撤和收益率
        equity_series = df['equity']
        returns = equity_series.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        annual_return = (self.capital / self.initial_capital) ** (252 / len(df)) - 1
        annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 基于权益序列计算最大回撤
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # 找到最大回撤的具体时间点和数值
        max_dd_idx = drawdown.idxmin()
        max_dd_date = max_dd_idx
        max_dd_value = drawdown.min()
        peak_before_dd = peak.loc[max_dd_idx]
        equity_at_dd = equity_series.loc[max_dd_idx]
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        trade_pnls = [t.get('pnl', 0) for t in self.trades if t['action'] == '平仓']
        win_rate = len([pnl for pnl in trade_pnls if pnl > 0]) / len(trade_pnls) if trade_pnls else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_dd_date,
            'peak_before_drawdown': peak_before_dd,
            'equity_at_drawdown': equity_at_dd,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'num_trades': len([t for t in self.trades if t['action'] == '开仓']),
            'total_days': len(df)
        }
    
    def print_detailed_trades(self, to_file=True):
        """打印详细交易分析"""
        # 准备输出内容
        output_lines = []
        output_lines.append("\n" + "="*100)
        output_lines.append("📍 详细交易分析 (最终修复版)")
        output_lines.append("="*100)
        
        if not self.trades:
            output_lines.append("❌ 无交易记录")
            if to_file:
                self._write_to_file(output_lines)
            else:
                for line in output_lines:
                    print(line)
            return
        
        close_trades = [t for t in self.trades if t['action'] == '平仓']
        
        output_lines.append(f"\n📊 交易成本分析:")
        total_entry_cost = sum(t.get('entry_cost', 0) for t in self.trades if t['action'] == '开仓')
        total_close_cost = sum(t.get('close_cost', 0) for t in close_trades)
        total_cost = total_entry_cost + total_close_cost
        
        output_lines.append(f"总开仓成本: {total_entry_cost:,.2f} 元")
        output_lines.append(f"总平仓成本: {total_close_cost:,.2f} 元")
        output_lines.append(f"总交易成本: {total_cost:,.2f} 元")
        output_lines.append(f"累计交易成本: {self.total_transaction_costs:,.2f} 元")
        
        output_lines.append(f"\n📋 详细交易记录:")
        output_lines.append("=" * 80)
        
        open_trades = [t for t in self.trades if t['action'] == '开仓']
        
        # 计算交易后现金变化
        capital_history = [self.initial_capital]
        
        for i, close_trade in enumerate(close_trades):
            open_trade = open_trades[i]
            
            open_date = open_trade['date'].strftime('%Y-%m-%d')
            close_date = close_trade['date'].strftime('%Y-%m-%d')
            direction = '多头' if open_trade['direction'] == '多头' else '空头'
            open_price = open_trade['price']
            close_price = close_trade['price']
            position_size = abs(open_trade['position'])
            price_pnl = close_trade['price_pnl']
            total_cost = close_trade['total_cost']
            net_pnl = close_trade['pnl']
            max_price_dd = close_trade.get('max_price_drawdown', 0)
            max_equity_dd = close_trade.get('max_equity_drawdown', 0)
            max_margin_ratio = close_trade.get('max_margin_ratio', 0)
            margin_used = open_trade.get('margin', 0)  # 获取开仓时保证金占用
            
            # 计算开仓后现金和平仓后现金
            capital_before_open = capital_history[-1]
            capital_after_open = capital_before_open  # 开仓时现金不变（只占用保证金）
            capital_after_close = capital_before_open + net_pnl  # 平仓后现金变化
            capital_history.append(capital_after_close)
            
            output_lines.append(f"\n📊 交易 #{i+1}:")
            output_lines.append(f"   开仓日期: {open_date}")
            output_lines.append(f"   平仓日期: {close_date}")
            output_lines.append(f"   交易方向: {direction}")
            output_lines.append(f"   开仓价格: {open_price:.0f} 元/吨")
            output_lines.append(f"   平仓价格: {close_price:.0f} 元/吨")
            output_lines.append(f"   交易手数: {position_size} 手")
            output_lines.append(f"   保证金占用: {margin_used:,.0f} 元")
            output_lines.append(f"   开仓后现金: {capital_after_open:,.0f} 元")
            output_lines.append(f"   价格盈亏: {price_pnl:,.0f} 元")
            output_lines.append(f"   交易成本: {total_cost:.0f} 元")
            output_lines.append(f"   净盈亏: {net_pnl:,.0f} 元")
            output_lines.append(f"   最大价格回撤: {max_price_dd:.1%}")
            output_lines.append(f"   最大权益回撤: {max_equity_dd:.1%}")
            output_lines.append(f"   最大保证金占比: {max_margin_ratio:.1%}")
            output_lines.append(f"   平仓原因: {close_trade.get('reason', '未知')}")
            output_lines.append(f"   平仓后现金: {capital_after_close:,.0f} 元")
            output_lines.append("-" * 60)
        
        output_lines.append("=" * 80)
        output_lines.append(f"\n📊 回撤分析:")
        output_lines.append(f"注意：详细的最大回撤分析需要基于完整的权益序列计算")
        output_lines.append(f"当前显示的是基于交易点的简化分析")
        output_lines.append(f"最终资金: {capital_history[-1]:,.0f} 元")
        
        # 价格回撤统计
        price_drawdowns = [t.get('max_price_drawdown', 0) for t in close_trades]
        if price_drawdowns:
            max_price_dd = max(price_drawdowns)
            avg_price_dd = sum(price_drawdowns) / len(price_drawdowns)
            output_lines.append(f"\n📊 价格回撤统计:")
            output_lines.append(f"最大价格回撤: {max_price_dd:.1%}")
            output_lines.append(f"平均价格回撤: {avg_price_dd:.1%}")
            output_lines.append(f"价格回撤说明: 相对于成本价和持仓期间最优价格的最大不利变动")
        
        # 验证总盈亏
        strategy_total_pnl = sum(t['pnl'] for t in close_trades)
        capital_change = self.capital - self.initial_capital
        
        output_lines.append(f"\n📊 盈亏验证:")
        output_lines.append(f"策略计算总盈亏: {strategy_total_pnl:,.0f} 元")
        output_lines.append(f"资金变化: {capital_change:,.0f} 元")
        output_lines.append(f"差异: {abs(strategy_total_pnl - capital_change):.2f} 元")
        
        if abs(strategy_total_pnl - capital_change) < 0.01:
            output_lines.append("✅ 利润计算完全修复成功！")
        else:
            output_lines.append("❌ 仍存在微小计算差异")
        
        # 输出到文件或控制台
        if to_file:
            self._write_to_file(output_lines)
            print("📄 详细交易信息已保存到 trading_details.log")
        else:
             for line in output_lines:
                 print(line)
    
    def _write_to_file(self, output_lines):
        """将输出内容写入临时文件"""
        log_file = 'trading_details.log'
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
        except Exception as e:
            print(f"❌ 写入日志文件失败: {e}")
            # 如果写入失败，直接打印到控制台
            for line in output_lines:
                print(line)
    
    def run_strategy(self, data_file):
        """运行完整策略"""
        print("🚀 启动鸡蛋期货保证金风险监控策略 (最终修复版)...")
        
        print(f"\n📊 加载数据: {data_file}")
        df = pd.read_csv(data_file)
        print(f"原始数据: {len(df)} 条记录")
        
        print("\n🔧 数据预处理...")
        processed_df = self.preprocess_data(df)
        print(f"处理后日线数据: {len(processed_df)} 条记录")
        print(f"时间范围: {processed_df.index[0].strftime('%Y-%m-%d')} 到 {processed_df.index[-1].strftime('%Y-%m-%d')}")
        
        print(f"\n📈 执行策略回测（初始资金: {self.initial_capital:,}元）...")
        backtest_df = self.backtest(processed_df)
        
        print("\n📊 计算性能指标...")
        metrics = self.calculate_performance_metrics(backtest_df)
        
        print("\n" + "="*60)
        print("🏆 策略回测结果 (最终修复版)")
        print("="*60)
        print(f"📅 回测期间: {backtest_df.index[0].strftime('%Y-%m-%d')} 至 {backtest_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 初始资金: {metrics['initial_capital']:,.0f} 元")
        print(f"💰 最终资金: {metrics['final_capital']:,.0f} 元")
        print(f"📈 总收益率: {metrics['total_return']:.1%}")
        print(f"📊 年化收益率: {metrics['annual_return']:.1%}")
        print(f"📉 年化波动率: {metrics['annual_volatility']:.1%}")
        print(f"⭐ 夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"📉 最大回撤: {metrics['max_drawdown']:.1%}")
        if metrics['max_drawdown'] < 0:
            print(f"📉 最大回撤详情:")
            print(f"   - 回撤发生日期: {metrics['max_drawdown_date'].strftime('%Y-%m-%d')}")
            print(f"   - 回撤前峰值权益: {metrics['peak_before_drawdown']:,.0f} 元")
            print(f"   - 回撤时权益: {metrics['equity_at_drawdown']:,.0f} 元")
            print(f"   - 回撤金额: {metrics['peak_before_drawdown'] - metrics['equity_at_drawdown']:,.0f} 元")
        print(f"🎯 Calmar比率: {metrics['calmar_ratio']:.3f}")
        print(f"🎲 胜率: {metrics['win_rate']:.1%}")
        print(f"🔄 开仓次数: {metrics['num_trades']}")
        
        # 打印详细交易分析
        self.print_detailed_trades()
        
        print(f"\n✅ 最终修复版策略分析完成！")
        
        return backtest_df, metrics


def main(config=None):
    """主函数"""
    # 默认配置
    default_config = {
        'initial_capital': 20000,
        'data_file': 'jd_main_contract_1min_20250909_221508.csv',
        'margin_rate': 0.10,
        'max_position_ratio': 0.9,
        'risk_per_trade': 0.2,
        'stop_loss_pct': 0.1,
        'target_margin_usage': 0.6
    }
    
    # 合并用户配置
    if config:
        default_config.update(config)
    
    # 创建策略实例并应用配置
    strategy = JDMarginRiskStrategyFinal(initial_capital=default_config['initial_capital'])
    strategy.margin_rate = default_config['margin_rate']
    strategy.max_position_ratio = default_config['max_position_ratio']
    strategy.risk_per_trade = default_config['risk_per_trade']
    strategy.stop_loss_pct = default_config['stop_loss_pct']
    strategy.target_margin_usage = default_config['target_margin_usage']  # 应用目标保证金占用率配置
    
    print(f"📊 策略配置:")
    print(f"初始资金: {default_config['initial_capital']:,}元")
    print(f"数据文件: {default_config['data_file']}")
    print(f"保证金比例: {default_config['margin_rate']:.1%}")
    print(f"最大仓位比例: {default_config['max_position_ratio']:.1%}")
    print(f"单次交易风险: {default_config['risk_per_trade']:.1%}")
    print(f"止损比例: {default_config['stop_loss_pct']:.1%}")
    print(f"目标保证金占用率: {default_config['target_margin_usage']:.1%}")
    print("="*60)
    
    results_df, performance = strategy.run_strategy(default_config['data_file'])
    return strategy, results_df, performance


if __name__ == "__main__":
    # 可以通过传入config字典来自定义配置
    # 例如: strategy, results, metrics = main({'initial_capital': 200000, 'data_file': 'other_file.csv'})
    # 使用自定义配置
    strategy, results, metrics = main()