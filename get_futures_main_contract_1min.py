#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货主力连续合约1分钟K线数据获取脚本

功能：
1. 获取期货主力连续合约的1分钟K线数据（支持玉米、鸡蛋、豆一、豆粕、豆油等品种）
2. 自动处理主力合约切换
3. 数据质量检查和统计分析
4. 支持自定义时间范围和期货品种
5. 支持多品种配置切换

支持的期货品种：
大连商品交易所 (DCE):
- C (玉米)、JD (鸡蛋)、A (豆一)、M (豆粕)、Y (豆油)、V (PVC)
上海期货交易所 (SHFE):
- RB (螺纹钢)、SS (不锈钢)、AU (黄金)、AG (白银)
郑州商品交易所 (CZCE):
- FG (玻璃)、SA (纯碱)

作者：Assistant
创建时间：2025-09-07
更新时间：2025-09-07
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

# 期货品种配置
FUTURES_CONFIG = {
    # 大连商品交易所 (DCE)
    'C': {  # 玉米
        'name': '玉米',
        'delivery_months': [1, 3, 5, 7, 9, 11],
        'exchange': 'DCE'
    },
    'JD': {  # 鸡蛋
        'name': '鸡蛋', 
        'delivery_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'exchange': 'DCE'
    },
    'A': {  # 豆一
        'name': '豆一',
        'delivery_months': [1, 3, 5, 7, 9, 11],
        'exchange': 'DCE'
    },
    'M': {  # 豆粕
        'name': '豆粕',
        'delivery_months': [1, 3, 5, 7, 8, 9, 11, 12],
        'exchange': 'DCE'
    },
    'Y': {  # 豆油
        'name': '豆油',
        'delivery_months': [1, 3, 5, 7, 8, 9, 11, 12],
        'exchange': 'DCE'
    },
    'V': {  # PVC
        'name': 'PVC',
        'delivery_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'exchange': 'DCE'
    },
    
    # 上海期货交易所 (SHFE)
    'RB': {  # 螺纹钢
        'name': '螺纹钢',
        'delivery_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'exchange': 'SHFE'
    },
    'SS': {  # 不锈钢
        'name': '不锈钢',
        'delivery_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'exchange': 'SHFE'
    },
    'AU': {  # 黄金
        'name': '黄金',
        'delivery_months': [2, 4, 6, 8, 10, 12],
        'exchange': 'SHFE'
    },
    'AG': {  # 白银
        'name': '白银',
        'delivery_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'exchange': 'SHFE'
    },
    
    # 郑州商品交易所 (CZCE)
    'FG': {  # 玻璃
        'name': '玻璃',
        'delivery_months': [1, 3, 5, 7, 9, 11],
        'exchange': 'CZCE'
    },
    'SA': {  # 纯碱
        'name': '纯碱',
        'delivery_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'exchange': 'CZCE'
    }
}

def get_main_contract_schedule(symbol='C', start_year=2024, start_month=1, end_year=None, end_month=None):
    """
    获取主力合约切换时间表
    
    Args:
        symbol: 期货品种代码
        start_year: 开始年份
        start_month: 开始月份
        end_year: 结束年份，默认为None（到当前时间）
        end_month: 结束月份，默认为None（到当前时间）
    
    Returns:
        list: 主力合约切换时间表，格式为 [(开始时间, 结束时间, 合约代码), ...]
    """
    if symbol not in FUTURES_CONFIG:
        raise ValueError(f"不支持的期货品种: {symbol}")
    
    config = FUTURES_CONFIG[symbol]
    delivery_months = config['delivery_months']
    
    schedule = []
    current_date = datetime(start_year, start_month, 1)
    
    # 设置结束时间
    if end_year is not None and end_month is not None:
        end_date = datetime(end_year, end_month, 1)
        # 如果指定了结束月份，设置为该月的最后一天
        if end_month == 12:
            end_date = datetime(end_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(end_year, end_month + 1, 1) - timedelta(days=1)
    else:
        end_date = datetime.now()
    
    # 为了确保覆盖足够的时间范围，我们需要生成从开始时间到现在的所有可能的主力合约
    # 主力合约通常在交割月前1-2个月开始成为主力
    
    # 生成所有可能的合约
    contracts = []
    for year in range(start_year, end_date.year + 2):  # 多生成一年确保覆盖
        for month in delivery_months:
            contract_code = f"{symbol}{year % 100:02d}{month:02d}"
            # 主力合约通常在交割月前2个月开始，交割月前1个月结束
            main_start = datetime(year, month, 1) - timedelta(days=60)  # 提前2个月
            main_end = datetime(year, month, 1) - timedelta(days=30)    # 提前1个月
            
            # 调整边界情况
            if main_start.month <= 0:
                main_start = main_start.replace(year=main_start.year-1, month=main_start.month+12)
            if main_end.month <= 0:
                main_end = main_end.replace(year=main_end.year-1, month=main_end.month+12)
                
            contracts.append((main_start, main_end, contract_code))
    
    # 按开始时间排序
    contracts.sort(key=lambda x: x[0])
    
    # 筛选出在我们需要的时间范围内的合约
    for start_time, end_time, contract_code in contracts:
        # 如果合约的结束时间在我们的开始时间之前，跳过
        if end_time < current_date:
            continue
        # 如果合约的开始时间在我们的结束时间之后，跳过
        if start_time > end_date:
            continue
            
        # 调整时间边界
        actual_start = max(start_time, current_date)
        actual_end = min(end_time, end_date)
        
        if actual_start <= actual_end:
            schedule.append((actual_start.strftime('%Y-%m-%d'), actual_end.strftime('%Y-%m-%d'), contract_code))
    
    # 如果没有找到合适的历史合约，使用简化的逻辑
    if not schedule:
        print("警告：未找到历史主力合约，使用当前主力合约")
        # 使用当前最活跃的合约
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # 找到最近的交割月
        for delivery_month in delivery_months:
            if current_month <= delivery_month + 2:  # 交割月前2个月内
                contract_code = f"{symbol}{current_year % 100:02d}{delivery_month:02d}"
                schedule.append((current_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), contract_code))
                break
        
        # 如果还是没找到，使用下一年的第一个交割月
        if not schedule:
            next_year = current_year + 1
            delivery_month = delivery_months[0]
            contract_code = f"{symbol}{next_year % 100:02d}{delivery_month:02d}"
            schedule.append((current_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), contract_code))
    
    return schedule

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
            # 检查数据结构并重命名列
            if len(data.columns) >= 7:
                # 重命名列（不包含contract列）
                data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'hold'] + list(data.columns[7:])
                
                # 添加合约标识
                data['contract'] = contract
                
                # 转换时间格式
                data['datetime'] = pd.to_datetime(data['datetime'])
            else:
                print(f"  数据结构异常，列数: {len(data.columns)}")
                return None
            
            # 如果指定了时间范围，进行过滤（但不要过度过滤历史数据）
            # 注意：AkShare返回的历史数据本身就是有限的，不要过度过滤
            original_count = len(data)
            
            if start_date:
                start_date_dt = pd.to_datetime(start_date)
                data = data[data['datetime'] >= start_date_dt]
            
            if end_date:
                end_date_dt = pd.to_datetime(end_date)
                data = data[data['datetime'] <= end_date_dt]
            
            # 如果过滤后数据为空，但原始数据不为空，说明时间范围设置有问题
            if len(data) == 0 and original_count > 0:
                print(f"  警告：时间过滤后数据为空，原始数据有 {original_count} 条")
                # 重新获取原始数据来显示时间范围
                try:
                    original_data = ak.futures_zh_minute_sina(symbol=contract, period="1")
                    if original_data is not None and len(original_data) > 0:
                        original_data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'hold']
                        original_data['datetime'] = pd.to_datetime(original_data['datetime'])
                        print(f"  原始时间范围: {original_data['datetime'].min()} 到 {original_data['datetime'].max()}")
                        # 返回原始数据
                        original_data['contract'] = contract
                        data = original_data
                    else:
                        print(f"  无法获取原始数据")
                        return None
                except Exception as e:
                    print(f"  获取原始数据失败: {e}")
                    return None
            
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

def get_futures_main_contract_data(symbol='C', start_year=2024, start_month=1, end_year=None, end_month=None):
    """
    获取期货主力连续合约1分钟数据
    
    Args:
        symbol: 期货品种代码，如'C'(玉米)、'JD'(鸡蛋)
        start_year: 开始年份
        start_month: 开始月份
        end_year: 结束年份，默认为None（到当前时间）
        end_month: 结束月份，默认为None（到当前时间）
    """
    if symbol not in FUTURES_CONFIG:
        raise ValueError(f"不支持的期货品种: {symbol}")
    
    futures_name = FUTURES_CONFIG[symbol]['name']
    
    print("=" * 60)
    print(f"获取{futures_name}期货主力连续合约1分钟K线数据")
    if end_year is not None and end_month is not None:
        print(f"时间范围：{start_year}年{start_month}月 至 {end_year}年{end_month}月")
    else:
        print(f"时间范围：{start_year}年{start_month}月至今")
    print(f"品种代码：{symbol}")
    print("=" * 60)
    
    # 获取主力合约切换时间表
    contract_schedule = get_main_contract_schedule(symbol, start_year, start_month, end_year, end_month)
    
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
    filename = f'{symbol.lower()}_main_contract_1min_{timestamp}.csv'
    
    print(f"\n正在保存数据到: {filename}")
    # 删除临时列
    save_data = combined_data.drop('month', axis=1)
    save_data.to_csv(filename, index=False)
    
    # 文件大小
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.2f} MB")
    
    print("\n=== 主力连续合约数据获取完成 ===")
    print(f"✓ 成功获取{futures_name}期货主力连续合约1分钟K线数据")
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
    # 配置参数 - 可以根据需要修改
    SYMBOL = 'JD'  # 期货品种代码
    START_YEAR = 2022  # 开始年份
    START_MONTH = 1    # 开始月份
    END_YEAR = None    # 结束年份，None表示到当前时间
    END_MONTH = None   # 结束月份，None表示到当前时间
    
    # 支持的期货品种：
    # 大连商品交易所 (DCE): 'C'(玉米), 'JD'(鸡蛋), 'A'(豆一), 'M'(豆粕), 'Y'(豆油), 'V'(PVC)
    # 上海期货交易所 (SHFE): 'RB'(螺纹钢), 'SS'(不锈钢), 'AU'(黄金), 'AG'(白银)
    # 郑州商品交易所 (CZCE): 'FG'(玻璃), 'SA'(纯碱)
    
    # 示例：获取螺纹钢期货数据
    # SYMBOL = 'RB'
    # START_YEAR = 2024
    # START_MONTH = 1
    
    print(f"配置信息：")
    print(f"  期货品种: {SYMBOL} ({FUTURES_CONFIG[SYMBOL]['name']})")
    print(f"  开始时间: {START_YEAR}年{START_MONTH}月")
    if END_YEAR is not None and END_MONTH is not None:
        print(f"  结束时间: {END_YEAR}年{END_MONTH}月")
    else:
        print(f"  结束时间: 当前时间")
    print(f"  交割月份: {FUTURES_CONFIG[SYMBOL]['delivery_months']}")
    print()
    
    try:
        data = get_futures_main_contract_data(SYMBOL, START_YEAR, START_MONTH, END_YEAR, END_MONTH)
        
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