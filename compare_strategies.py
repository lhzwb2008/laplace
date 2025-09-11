#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比策略分析：主力合约 vs 多合约换仓
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_indicators(df):
    """计算技术指标"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 价格位置
    rolling_min = df['close'].rolling(window=30).min()
    rolling_max = df['close'].rolling(window=30).max()
    df['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)
    
    return df

def simulate_main_contract_only(df):
    """只用主力合约，不换仓"""
    # 筛选主力合约数据
    main_data = df[df['contract'] == df['main_contract']].copy()
    main_data = main_data.drop_duplicates(subset=['datetime']).sort_values('datetime')
    main_data = main_data.set_index('datetime')
    
    # 计算指标
    main_data = calculate_indicators(main_data)
    main_data['month'] = main_data.index.month
    
    # 初始化
    capital = 20000
    position = 0
    trades = []
    open_price = 0  # 记录开仓价格
    
    # 回测
    for idx, row in main_data.iterrows():
        if pd.isna(row['rsi']) or pd.isna(row['price_position']):
            continue
            
        month = row['month']
        price = row['close']
        
        # 买入信号
        if month in [6, 7, 8] and position <= 0:
            if row['price_position'] <= 0.7 and row['rsi'] <= 80:
                if position < 0:  # 先平空
                    profit = -position * (open_price - price) * 10
                    capital += profit
                    trades.append({
                        'date': idx,
                        'action': '平空',
                        'price': price,
                        'profit': profit
                    })
                
                # 开多
                position_size = min(2, int(capital / (price * 10 * 0.15)))
                position = position_size
                open_price = price
                trades.append({
                    'date': idx,
                    'action': '开多',
                    'price': price,
                    'position': position
                })
        
        # 卖出信号
        elif month in [11, 12, 1] and position >= 0:
            if row['price_position'] >= 0.5 and row['rsi'] >= 20:
                if position > 0:  # 先平多
                    profit = position * (price - open_price) * 10
                    capital += profit
                    trades.append({
                        'date': idx,
                        'action': '平多',
                        'price': price,
                        'profit': profit
                    })
                
                # 开空
                position_size = min(2, int(capital / (price * 10 * 0.15)))
                position = -position_size
                open_price = price
                trades.append({
                    'date': idx,
                    'action': '开空',
                    'price': price,
                    'position': position
                })
    
    # 最后平仓
    if position != 0:
        last_price = main_data.iloc[-1]['close']
        if position > 0:
            profit = position * (last_price - open_price) * 10
        else:
            profit = -position * (open_price - last_price) * 10
        capital += profit
        trades.append({
            'date': main_data.index[-1],
            'action': '最终平仓',
            'price': last_price,
            'profit': profit
        })
    
    return capital, trades

def main():
    """主函数"""
    print("📊 策略对比分析：主力合约 vs 多合约换仓")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_csv('jd_all_contracts_daily_2022-2025_20250911_145111.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 1. 只用主力合约（不换仓）
    print("\n🎯 策略1：只用主力合约（不换仓）")
    final_capital_main, trades_main = simulate_main_contract_only(df)
    
    initial_capital = 20000
    profit_main = final_capital_main - initial_capital
    return_main = profit_main / initial_capital * 100
    
    print(f"  初始资金: {initial_capital:,}元")
    print(f"  最终资金: {final_capital_main:,.0f}元")
    print(f"  收益: {profit_main:,.0f}元")
    print(f"  收益率: {return_main:.1f}%")
    print(f"  交易次数: {len(trades_main)}次")
    
    # 2. 多合约换仓策略（从实际运行结果）
    print("\n🔄 策略2：多合约换仓")
    final_capital_rollover = 18001
    rollover_cost = 9536
    
    profit_rollover = final_capital_rollover - initial_capital
    return_rollover = profit_rollover / initial_capital * 100
    
    print(f"  初始资金: {initial_capital:,}元")
    print(f"  最终资金: {final_capital_rollover:,}元")
    print(f"  收益: {profit_rollover:,}元")
    print(f"  收益率: {return_rollover:.1f}%")
    print(f"  换仓成本: {rollover_cost:,}元")
    print(f"  换仓次数: 27次")
    
    # 3. 对比分析
    print("\n📈 对比分析:")
    print(f"  收益率差异: {return_main - return_rollover:.1f}%")
    print(f"  换仓成本影响: -{rollover_cost/initial_capital*100:.1f}%")
    
    # 如果没有换仓成本
    capital_without_cost = final_capital_rollover + rollover_cost
    return_without_cost = (capital_without_cost - initial_capital) / initial_capital * 100
    print(f"  多合约策略无换仓成本收益率: {return_without_cost:.1f}%")
    
    print("\n💡 结论:")
    if return_main > return_rollover:
        print(f"  ✅ 只用主力合约策略更优，超额收益{return_main - return_rollover:.1f}%")
    else:
        print(f"  ❌ 多合约换仓策略更优，超额收益{return_rollover - return_main:.1f}%")
    
    print(f"\n  换仓成本是策略失效的主要原因！")
    print(f"  换仓成本占初始资金的{rollover_cost/initial_capital*100:.1f}%")

if __name__ == "__main__":
    main()
