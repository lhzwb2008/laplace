#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取玉米期货主力连续合约1分钟K线数据（从2024年9月开始）
通过主力合约切换时间获取对应合约数据并拼接
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import time
warnings.filterwarnings('ignore')

try:
    import akshare as ak
except ImportError:
    print("错误：未安装akshare")
    print("请先安装：pip install akshare")
    exit(1)

def get_main_contract_schedule():
    """
    定义玉米期货主力合约切换时间表
    玉米期货交割月份：1, 3, 5, 7, 9, 11月
    主力合约通常在交割月前1-2个月切换
    """
    # 主力合约切换时间表 (大致时间，实际可能有微调)
    contract_schedule = [
        # (开始时间, 结束时间, 合约代码)
        ('2024-09-01', '2024-10-31', 'C2411'),  # 2024年9-10月主力：C2411
        ('2024-11-01', '2024-12-31', 'C2501'),  # 2024年11-12月主力：C2501
        ('2025-01-01', '2025-02-28', 'C2503'),  # 2025年1-2月主力：C2503
        ('2025-03-01', '2025-04-30', 'C2505'),  # 2025年3-4月主力：C2505
        ('2025-05-01', '2025-06-30', 'C2507'),  # 2025年5-6月主力：C2507
        ('2025-07-01', '2025-08-31', 'C2509'),  # 2025年7-8月主力：C2509
        ('2025-09-01', '2025-09-30', 'C2511'),  # 2025年9月主力：C2511
    ]
    
    return contract_schedule

def get_contract_minute_data(contract, start_date=None, end_date=None):
    """
    获取单个合约的1分钟数据
    """
    try:
        print(f"正在获取合约 {contract} 的数据...")
        
        # 使用AkShare获取1分钟数据
        data = ak.futures_zh_minute_sina(
            symbol=contract, 
            period="1"
        )
        
        if data is not None and len(data) > 0:
            # 添加合约标识
            data['contract'] = contract
            
            # 重命名列
            data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'hold', 'contract']
            
            # 转换时间格式
            data['datetime'] = pd.to_datetime(data['datetime'])
            
            # 如果指定了时间范围，进行过滤
            if start_date:
                start_date = pd.to_datetime(start_date)
                data = data[data['datetime'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                data = data[data['datetime'] <= end_date]
            
            print(f"合约 {contract}: 获取到 {len(data)} 条数据")
            if len(data) > 0:
                print(f"  时间范围: {data['datetime'].min()} 到 {data['datetime'].max()}")
            
            return data
        else:
            print(f"合约 {contract}: 无数据")
            return None
            
    except Exception as e:
        print(f"获取合约 {contract} 数据失败: {e}")
        return None

def get_corn_main_contract_data():
    """
    获取玉米期货主力连续合约1分钟数据
    """
    print("=" * 60)
    print("获取玉米期货主力连续合约1分钟K线数据")
    print("时间范围：2024年9月至今")
    print("=" * 60)
    
    # 获取主力合约切换时间表
    contract_schedule = get_main_contract_schedule()
    
    print("主力合约切换时间表:")
    for start_date, end_date, contract in contract_schedule:
        print(f"  {start_date} ~ {end_date}: {contract}")
    print()
    
    all_data = []
    successful_contracts = []
    
    for start_date, end_date, contract in contract_schedule:
        print(f"\n=== 获取 {contract} 数据 ({start_date} ~ {end_date}) ===")
        
        data = get_contract_minute_data(contract, start_date, end_date)
        
        if data is not None and len(data) > 0:
            all_data.append(data)
            successful_contracts.append(contract)
            
            # 显示数据统计
            print(f"  数据条数: {len(data):,}")
            print(f"  价格范围: {data['close'].min():.2f} ~ {data['close'].max():.2f}")
            print(f"  平均成交量: {data['volume'].mean():.0f}")
        
        # 添加延时避免请求过快
        time.sleep(2)
    
    if not all_data:
        print("\n未获取到任何数据")
        return None
    
    # 合并所有数据
    print("\n=== 合并数据 ===")
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 按时间排序
    combined_data = combined_data.sort_values('datetime')
    combined_data = combined_data.reset_index(drop=True)
    
    # 数据质量检查
    print(f"\n总数据条数: {len(combined_data):,}")
    print(f"成功获取的合约: {', '.join(successful_contracts)}")
    print(f"时间范围: {combined_data['datetime'].min()} 到 {combined_data['datetime'].max()}")
    
    # 检查缺失值
    missing_data = combined_data.isnull().sum()
    print("\n缺失值检查:")
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"  {col}: {missing} 个缺失值")
        else:
            print(f"  {col}: 无缺失值")
    
    # 基本统计
    print("\n价格统计:")
    print(f"  最高价: {combined_data['high'].max():.2f}")
    print(f"  最低价: {combined_data['low'].min():.2f}")
    print(f"  平均收盘价: {combined_data['close'].mean():.2f}")
    
    print(f"\n成交量统计:")
    print(f"  总成交量: {combined_data['volume'].sum():,}")
    print(f"  平均成交量: {combined_data['volume'].mean():.0f}")
    
    # 合约数据分布
    print("\n合约数据分布:")
    contract_counts = combined_data['contract'].value_counts()
    for contract, count in contract_counts.items():
        percentage = count / len(combined_data) * 100
        print(f"  {contract}: {count:,} 条 ({percentage:.1f}%)")
    
    # 按月份统计
    print("\n按月份统计:")
    combined_data['month'] = combined_data['datetime'].dt.to_period('M')
    monthly_counts = combined_data['month'].value_counts().sort_index()
    for month, count in monthly_counts.items():
        percentage = count / len(combined_data) * 100
        print(f"  {month}: {count:,} 条 ({percentage:.1f}%)")
    
    # 检查数据连续性
    print("\n数据连续性检查:")
    time_diff = combined_data['datetime'].diff()
    gaps = time_diff[time_diff > pd.Timedelta(minutes=5)]  # 超过5分钟的间隔
    if len(gaps) > 0:
        print(f"  发现 {len(gaps)} 个时间间隔超过5分钟的数据缺口")
        print("  主要缺口:")
        for i, (idx, gap) in enumerate(gaps.head(5).items()):
            prev_time = combined_data.loc[idx-1, 'datetime']
            curr_time = combined_data.loc[idx, 'datetime']
            print(f"    {prev_time} -> {curr_time} (间隔: {gap})")
    else:
        print("  数据连续性良好，无明显缺口")
    
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'corn_main_contract_1min_{timestamp}.csv'
    
    print(f"\n正在保存数据到: {filename}")
    # 删除临时列
    save_data = combined_data.drop('month', axis=1)
    save_data.to_csv(filename, index=False)
    
    # 文件大小
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    print("\n=== 主力连续合约数据获取完成 ===")
    print(f"✓ 成功获取玉米期货主力连续合约1分钟K线数据")
    print(f"✓ 数据条数: {len(combined_data):,}")
    print(f"✓ 涉及合约: {len(successful_contracts)} 个")
    print(f"✓ 时间跨度: 约 {(combined_data['datetime'].max() - combined_data['datetime'].min()).days} 天")
    print(f"✓ 保存文件: {filename}")
    
    # 显示前几行数据
    print("\n前5行数据预览:")
    print(save_data.head())
    
    print("\n数据特点:")
    print("1. 主力连续合约数据，已处理合约切换")
    print("2. 1分钟级别的高频数据")
    print("3. 包含完整的OHLCV数据")
    print("4. 时间跨度覆盖过去一年")
    
    print("\n使用建议:")
    print("1. 适合长期策略回测")
    print("2. 主力合约连续性分析")
    print("3. 技术指标计算")
    print("4. 量价关系研究")
    print("5. 季节性规律分析")
    
    print("\n注意事项:")
    print("1. 数据已按主力合约切换时间拼接")
    print("2. 合约切换点可能存在价格跳跃")
    print("3. 建议结合基本面分析")
    print("4. 注意交易时间和节假日影响")
    
    return combined_data

def main():
    """
    主函数
    """
    try:
        data = get_corn_main_contract_data()
        
        if data is not None:
            print("\n数据获取成功！")
            print("可以开始进行量化分析了。")
            return True
        else:
            print("\n数据获取失败，请检查网络连接和参数设置。")
            return False
            
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n程序执行完成！")
    else:
        print("\n程序执行失败！")