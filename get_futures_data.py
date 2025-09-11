#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货数据获取工具

功能：
1. 获取指定期货品种所有合约的K线数据（支持1分钟、日线）
2. 包括主力和非主力合约
3. 支持自定义时间范围
4. 数据质量检查和统计分析
5. 自动识别主力合约切换

作者：Assistant
创建时间：2025-01-09
更新时间：2025-01-09
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

class FuturesDataFetcher:
    """期货数据获取器"""
    
    def __init__(self):
        self.supported_symbols = {
            'JD': '鸡蛋',
            'RB': '螺纹钢', 
            'HC': '热轧卷板',
            'I': '铁矿石',
            'J': '焦炭',
            'JM': '焦煤',
            'A': '豆一',
            'M': '豆粕',
            'Y': '豆油',
            'C': '玉米',
            'CS': '玉米淀粉',
            'P': '棕榈油',
            'V': 'PVC',
            'PP': 'PP',
            'L': 'LLDPE',
            'TA': 'PTA',
            'MA': '甲醇',
            'FG': '玻璃',
            'SA': '纯碱',
            'UR': '尿素'
        }
        
    def generate_contract_codes(self, symbol, start_year, end_year):
        """生成合约代码列表"""
        contracts = []
        
        for year in range(start_year, end_year + 1):
            year_suffix = str(year)[-2:]  # 取年份后两位
            
            # 生成12个月的合约
            for month in range(1, 13):
                contract_code = f"{symbol}{year_suffix}{month:02d}"
                contracts.append(contract_code)
        
        return contracts
    
    def fetch_single_contract(self, contract_code, period='1', retries=3):
        """
        获取单个合约的K线数据
        
        Args:
            contract_code: 合约代码，如 'JD2501'
            period: 数据周期 '1'=1分钟, 'daily'=日线
            retries: 重试次数
        """
        for attempt in range(retries):
            try:
                if period == 'daily':
                    # 获取日线数据
                    df = ak.futures_zh_daily_sina(symbol=contract_code)
                else:
                    # 获取1分钟数据
                    df = ak.futures_zh_minute_sina(symbol=contract_code, period=period)
                
                if df is None or len(df) == 0:
                    return None
                
                # 标准化列名
                if 'datetime' not in df.columns and 'date' in df.columns:
                    df['datetime'] = df['date']
                
                # 添加合约标识
                df['contract'] = contract_code
                
                # 确保必要的列存在
                required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'hold']
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'hold':
                            df[col] = 0  # 如果没有持仓量数据，设为0
                        else:
                            print(f"警告：合约 {contract_code} 缺少列 {col}")
                            return None
                
                return df[required_cols + ['contract']]
                
            except Exception as e:
                print(f"获取合约 {contract_code} 数据失败 (尝试 {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    time.sleep(1)  # 等待1秒后重试
                
        return None
    
    def get_all_contracts_data(self, symbol='JD', start_year=2024, end_year=2025, 
                              start_date=None, end_date=None, period='1'):
        """
        获取指定品种所有合约的K线数据
        
        Args:
            symbol: 期货品种代码，如 'JD'
            start_year: 开始年份
            end_year: 结束年份  
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            period: 数据周期 '1'=1分钟, 'daily'=日线
        """
        print(f"📊 开始获取 {symbol}({self.supported_symbols.get(symbol, '未知品种')}) "
              f"{start_year}-{end_year}年 {period}周期 数据...")
        
        # 生成合约代码
        contracts = self.generate_contract_codes(symbol, start_year, end_year)
        print(f"生成 {len(contracts)} 个合约代码")
        
        all_data = []
        successful_contracts = []
        failed_contracts = []
        
        for i, contract in enumerate(contracts, 1):
            print(f"[{i:3d}/{len(contracts)}] 获取 {contract} 数据...", end=' ')
            
            df = self.fetch_single_contract(contract, period)
            
            if df is not None and len(df) > 0:
                # 日期过滤
                if start_date or end_date:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    if start_date:
                        df = df[df['datetime'] >= start_date]
                    if end_date:
                        df = df[df['datetime'] <= end_date]
                
                if len(df) > 0:
                    all_data.append(df)
                    successful_contracts.append(contract)
                    print(f"✅ {len(df)} 条记录")
                else:
                    failed_contracts.append(contract)
                    print("❌ 日期过滤后无数据")
            else:
                failed_contracts.append(contract)
                print("❌ 获取失败")
            
            # 避免请求过于频繁
            time.sleep(0.1)
        
        if not all_data:
            print("❌ 没有获取到任何数据")
            return None
        
        # 合并所有数据
        print("\n📦 合并数据...")
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
        combined_df = combined_df.sort_values(['datetime', 'contract']).reset_index(drop=True)
        
        # 识别主力合约
        print("🔍 识别主力合约...")
        combined_df = self.identify_main_contract(combined_df, period)
        
        # 数据统计
        print(f"\n📈 数据获取完成:")
        print(f"  总记录数: {len(combined_df):,} 条")
        print(f"  时间范围: {combined_df['datetime'].min()} 到 {combined_df['datetime'].max()}")
        print(f"  成功合约: {len(successful_contracts)} 个")
        print(f"  失败合约: {len(failed_contracts)} 个")
        
        if successful_contracts:
            print(f"  成功合约: {', '.join(successful_contracts[:10])}" + 
                  (f" 等{len(successful_contracts)}个" if len(successful_contracts) > 10 else ""))
        
        if failed_contracts:
            print(f"  失败合约: {', '.join(failed_contracts[:10])}" + 
                  (f" 等{len(failed_contracts)}个" if len(failed_contracts) > 10 else ""))
        
        return combined_df
    
    def identify_main_contract(self, df, period):
        """识别主力合约"""
        df = df.copy()
        
        # 根据周期确定分组方式
        if period == 'daily':
            # 日线数据：按日期分组
            df['date'] = df['datetime'].dt.date
            daily_volume = df.groupby(['date', 'contract'])['volume'].sum().reset_index()
        else:
            # 分钟数据：按日期分组
            df['date'] = df['datetime'].dt.date
            daily_volume = df.groupby(['date', 'contract'])['volume'].sum().reset_index()
        
        # 找出每日成交量最大的合约作为主力合约
        main_contracts = daily_volume.loc[daily_volume.groupby('date')['volume'].idxmax()]
        main_contracts = main_contracts[['date', 'contract']].rename(columns={'contract': 'main_contract'})
        
        # 合并主力合约信息
        df = df.merge(main_contracts, on='date', how='left')
        
        # 统计主力合约切换
        main_switches = main_contracts['main_contract'].ne(main_contracts['main_contract'].shift()).sum()
        print(f"  主力合约切换: {main_switches} 次")
        
        return df
    
    def save_data(self, df, symbol, period, start_year, end_year):
        """保存数据到CSV文件"""
        if df is None or len(df) == 0:
            print("❌ 没有数据可保存")
            return None
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        period_suffix = "daily" if period == 'daily' else "1min"
        filename = f"{symbol.lower()}_all_contracts_{period_suffix}_{start_year}-{end_year}_{timestamp}.csv"
        
        # 保存文件
        df.to_csv(filename, index=False)
        print(f"💾 数据已保存到: {filename}")
        
        return filename

def main():
    """主函数"""
    # 配置参数
    CONFIG = {
        'SYMBOL': 'JD',           # 期货品种
        'START_YEAR': 2022,       # 开始年份
        'END_YEAR': 2025,         # 结束年份
        'START_DATE': None,       # 开始日期 'YYYY-MM-DD' 或 None
        'END_DATE': None,         # 结束日期 'YYYY-MM-DD' 或 None
        'PERIOD': 'daily',        # 数据周期: '1'=1分钟, 'daily'=日线
        'SAVE_FILE': True         # 是否保存文件
    }
    
    print("🚀 期货数据获取工具")
    print("=" * 50)
    
    # 创建数据获取器
    fetcher = FuturesDataFetcher()
    
    # 显示配置
    period_name = "日线" if CONFIG['PERIOD'] == 'daily' else "1分钟"
    print(f"📋 配置信息:")
    print(f"  品种: {CONFIG['SYMBOL']} ({fetcher.supported_symbols.get(CONFIG['SYMBOL'], '未知')})")
    print(f"  年份: {CONFIG['START_YEAR']}-{CONFIG['END_YEAR']}")
    print(f"  周期: {period_name}")
    if CONFIG['START_DATE']:
        print(f"  开始日期: {CONFIG['START_DATE']}")
    if CONFIG['END_DATE']:
        print(f"  结束日期: {CONFIG['END_DATE']}")
    print()
    
    # 获取数据
    df = fetcher.get_all_contracts_data(
        symbol=CONFIG['SYMBOL'],
        start_year=CONFIG['START_YEAR'],
        end_year=CONFIG['END_YEAR'],
        start_date=CONFIG['START_DATE'],
        end_date=CONFIG['END_DATE'],
        period=CONFIG['PERIOD']
    )
    
    if df is not None and CONFIG['SAVE_FILE']:
        filename = fetcher.save_data(
            df, CONFIG['SYMBOL'], CONFIG['PERIOD'], 
            CONFIG['START_YEAR'], CONFIG['END_YEAR']
        )
        
        # 显示数据样例
        print(f"\n📋 数据样例（前5行）:")
        print(df.head().to_string())
        
        print(f"\n✅ 获取完成！数据文件：{filename}")
    
    return df

if __name__ == "__main__":
    data = main()
