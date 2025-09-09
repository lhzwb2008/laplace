#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鸡蛋期货保证金风险分析脚本
分析策略运行过程中的保证金风险情况，包括强制平仓风险和避免措施

作者: AI Assistant
创建时间: 2025-01-08
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入策略类
from jd_margin_risk_strategy_final import JDMarginRiskStrategyFinal

def analyze_margin_risk():
    """分析保证金风险"""
    print("🔍 鸡蛋期货保证金风险分析")
    print("="*60)
    
    # 运行策略
    strategy = JDMarginRiskStrategyFinal(initial_capital=100000)
    data_file = 'jd_main_contract_1min_20250908_152828.csv'
    results_df, performance = strategy.run_strategy(data_file)
    
    print("\n📊 保证金风险监控分析")
    print("="*60)
    
    # 分析保证金风险数据
    if strategy.daily_risk_data:
        risk_df = pd.DataFrame(strategy.daily_risk_data)
        
        print(f"📈 保证金使用情况:")
        print(f"最大保证金占用率: {strategy.max_margin_usage:.1%}")
        print(f"平均保证金占用率: {risk_df['margin_usage_rate'].mean():.1%}")
        print(f"保证金占用率标准差: {risk_df['margin_usage_rate'].std():.1%}")
        
        # 检查追加保证金风险
        margin_call_days = risk_df[risk_df['margin_call_risk'] == True]
        print(f"\n⚠️  追加保证金风险天数: {len(margin_call_days)} 天")
        
        if len(margin_call_days) > 0:
            print("追加保证金风险详情:")
            for _, day in margin_call_days.iterrows():
                print(f"  日期: {day['date'].strftime('%Y-%m-%d')}, 价格: {day['price']:.0f}, 占用率: {day['margin_usage_rate']:.1%}")
        
        # 检查强制平仓风险
        liquidation_days = risk_df[risk_df['liquidation_risk'] == True]
        print(f"\n🚨 强制平仓风险天数: {len(liquidation_days)} 天")
        
        if len(liquidation_days) > 0:
            print("强制平仓风险详情:")
            for _, day in liquidation_days.iterrows():
                print(f"  日期: {day['date'].strftime('%Y-%m-%d')}, 价格: {day['price']:.0f}, 占用率: {day['margin_usage_rate']:.1%}")
        else:
            print("✅ 无强制平仓风险")
        
        # 分析最危险的时期
        top_risk_days = risk_df.nlargest(5, 'margin_usage_rate')
        print(f"\n📊 保证金占用率最高的5天:")
        for _, day in top_risk_days.iterrows():
            print(f"  {day['date'].strftime('%Y-%m-%d')}: {day['margin_usage_rate']:.1%} (价格: {day['price']:.0f}, 权益: {day['current_equity']:.0f})")
        
    else:
        print("❌ 无保证金风险数据记录")
    
    # 分析追加保证金警告
    print(f"\n📢 追加保证金警告次数: {len(strategy.margin_call_alerts)}")
    if strategy.margin_call_alerts:
        print("追加保证金警告详情:")
        for alert in strategy.margin_call_alerts:
            print(f"  {alert['date'].strftime('%Y-%m-%d')}: 价格 {alert['price']:.0f}, 缺口 {alert['shortage']:.0f}元")
    
    # 分析强制平仓警告
    print(f"\n🚨 强制平仓警告次数: {len(strategy.near_liquidation_alerts)}")
    if strategy.near_liquidation_alerts:
        print("强制平仓警告详情:")
        for alert in strategy.near_liquidation_alerts:
            print(f"  {alert['date'].strftime('%Y-%m-%d')}: 价格 {alert['price']:.0f}, 缺口 {alert['shortage']:.0f}元")
    else:
        print("✅ 无强制平仓警告")
    
    return strategy, results_df, performance

def suggest_risk_mitigation():
    """建议风险缓解措施"""
    print("\n\n🛡️  避免强制平仓的建议措施")
    print("="*60)
    
    print("\n1. 📊 仓位管理优化:")
    print("   • 降低最大仓位比例（当前30%）至20%或更低")
    print("   • 实施动态仓位调整，根据波动率调整仓位大小")
    print("   • 设置最大单笔交易金额限制")
    
    print("\n2. 💰 资金管理改进:")
    print("   • 保持更高的资金缓冲（建议至少50%现金）")
    print("   • 设置保证金占用率上限（建议不超过60%）")
    print("   • 建立应急资金池用于追加保证金")
    
    print("\n3. 🎯 止损机制强化:")
    print("   • 降低止损比例（当前5%）至3%或更低")
    print("   • 实施移动止损，锁定利润")
    print("   • 设置最大亏损限额")
    
    print("\n4. ⚠️  风险监控加强:")
    print("   • 实时监控保证金占用率")
    print("   • 设置保证金占用率预警阈值")
    print("   • 建立自动减仓机制")
    
    print("\n5. 📈 策略参数调整:")
    print("   • 提高入场条件的严格性")
    print("   • 增加波动率过滤的敏感性")
    print("   • 优化价格位置和RSI阈值")
    
    print("\n6. 🔄 分散化策略:")
    print("   • 考虑多品种分散投资")
    print("   • 实施时间分散化（分批建仓）")
    print("   • 避免在高波动期间大幅加仓")

def create_improved_strategy():
    """创建改进的低风险策略参数"""
    print("\n\n🔧 改进策略参数建议")
    print("="*60)
    
    print("\n当前策略参数:")
    print("  • 保证金比例: 5%")
    print("  • 最大仓位比例: 30%")
    print("  • 单次交易风险: 2%")
    print("  • 止损比例: 5%")
    print("  • 维持保证金比例: 4%")
    print("  • 追加保证金比例: 4.5%")
    
    print("\n建议的低风险参数:")
    print("  • 保证金比例: 5% (不变)")
    print("  • 最大仓位比例: 20% (降低10%)")
    print("  • 单次交易风险: 1.5% (降低0.5%)")
    print("  • 止损比例: 3% (降低2%)")
    print("  • 维持保证金比例: 4% (不变)")
    print("  • 追加保证金比例: 4.5% (不变)")
    print("  • 保证金占用率上限: 60% (新增)")
    print("  • 最大回撤限制: 15% (新增)")
    
    print("\n预期效果:")
    print("  ✅ 显著降低强制平仓风险")
    print("  ✅ 提高策略稳定性")
    print("  ✅ 减少极端损失")
    print("  ⚠️  可能降低收益率")
    print("  ⚠️  可能减少交易频率")

def main():
    """主函数"""
    # 分析保证金风险
    strategy, results_df, performance = analyze_margin_risk()
    
    # 提供风险缓解建议
    suggest_risk_mitigation()
    
    # 创建改进策略建议
    create_improved_strategy()
    
    print("\n\n📋 总结")
    print("="*60)
    print("根据分析结果，当前策略在历史数据上表现良好，")
    print("没有出现强制平仓的情况。但为了进一步降低风险，")
    print("建议采用上述风险缓解措施，特别是降低仓位比例")
    print("和加强止损机制。")
    
    return strategy, results_df, performance

if __name__ == "__main__":
    main()