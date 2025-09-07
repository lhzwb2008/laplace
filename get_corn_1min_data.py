#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取玉米期货1分钟K线历史数据（一年）
使用AkShare的futures_zh_minute_sina接口
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

def get_corn_contracts():
    """
    生成玉米期货合约代码
    """
    current_year = datetime.now().year
    delivery_months = [1, 3, 5, 7, 9, 11]  # 玉米期货主要交割月份
    contracts = []
    
    # 生成当前年份和下一年的合约
    for year in [current_year, current_year + 1]:
        for month in delivery_months:
            contract_code = f"C{str(year)[-2:]}{month:02d}"
            contracts.append(contract_code)
    
    return contracts[:6]  # 返回前6个合约

def get_minute_data_for_contract(contract):
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
            
            print(f"合约 {contract}: 获取到 {len(data)} 条数据")
            return data
        else:
            print(f"合约 {contract}: 无数据")
            return None
            
    except Exception as e:
        print(f"获取合约 {contract} 数据失败: {e}")
        return None

def get_corn_yearly_minute_data():
    """
    获取玉米期货一年的1分钟K线数据
    """
    print("=" * 60)
    print("获取玉米期货一年1分钟K线数据")
    print("=" * 60)
    
    # 获取合约列表
    contracts = get_corn_contracts()
    print(f"准备获取合约: {', '.join(contracts)}")
    
    all_data = []
    successful_contracts = []
    
    for contract in contracts:
        data = get_minute_data_for_contract(contract)
        if data is not None:
            all_data.append(data)
            successful_contracts.append(contract)
        
        # 添加延时避免请求过快
        time.sleep(1)
    
    if not all_data:
        print("未获取到任何数据")
        return None
    
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 数据处理
    print("\n=== 数据处理 ===")
    
    # 转换时间格式
    combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])
    
    # 按时间排序
    combined_data = combined_data.sort_values('datetime')
    
    # 数据质量检查
    print(f"总数据条数: {len(combined_data):,}")
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
    
    # 保存数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'corn_1min_kline_{timestamp}.csv'
    
    print(f"\n正在保存数据到: {filename}")
    combined_data.to_csv(filename, index=False)
    
    # 文件大小
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    print("\n=== 数据获取完成 ===")
    print(f"✓ 成功获取玉米期货1分钟K线数据")
    print(f"✓ 数据条数: {len(combined_data):,}")
    print(f"✓ 涉及合约: {len(successful_contracts)} 个")
    print(f"✓ 保存文件: {filename}")
    
    # 显示前几行数据
    print("\n前5行数据预览:")
    print(combined_data.head())
    
    print("\n数据特点:")
    print("1. 1分钟级别的高频数据")
    print("2. 包含开高低收价格信息")
    print("3. 包含成交量和持仓量数据")
    print("4. 适合短期交易策略分析")
    
    print("\n使用建议:")
    print("1. 适合日内交易策略回测")
    print("2. 技术指标计算(如MACD、RSI等)")
    print("3. 价格波动分析")
    print("4. 高频交易模型训练")
    print("5. 市场微观结构研究")
    
    print("\n注意事项:")
    print("1. 分钟数据时间范围相对较短")
    print("2. 数据包含多个合约")
    print("3. 需要考虑合约切换的影响")
    print("4. 建议结合日频数据进行长期分析")
    
    return combined_data

def main():
    """
    主函数
    """
    try:
        data = get_corn_yearly_minute_data()
        
        if data is not None:
            print("\n数据获取成功！")
            print("可以开始进行量化分析了。")
            return True
        else:
            print("\n数据获取失败，请检查网络连接和参数设置。")
            return False
            
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n程序执行完成！")
    else:
        print("\n程序执行失败！")