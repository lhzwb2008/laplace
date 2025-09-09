#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试期货数据获取脚本的endtime配置功能
"""

from get_futures_main_contract_1min import get_futures_main_contract_data

def test_endtime_config():
    """
    测试endtime配置功能
    """
    print("=== 测试期货数据获取脚本的endtime配置功能 ===")
    
    # 测试1: 获取2024年1-6月的数据
    print("\n测试1: 获取2024年1-6月的JD期货数据")
    data1 = get_futures_main_contract_data('JD', 2024, 1, 2024, 6)
    print(f"获取到数据: {len(data1)} 条")
    print(f"时间范围: {data1['datetime'].min()} 到 {data1['datetime'].max()}")
    
    # 测试2: 获取2024年7月至今的数据
    print("\n测试2: 获取2024年7月至今的JD期货数据")
    data2 = get_futures_main_contract_data('JD', 2024, 7)
    print(f"获取到数据: {len(data2)} 条")
    print(f"时间范围: {data2['datetime'].min()} 到 {data2['datetime'].max()}")
    
    # 测试3: 获取2024年全年数据
    print("\n测试3: 获取2024年全年的JD期货数据")
    data3 = get_futures_main_contract_data('JD', 2024, 1, 2024, 12)
    print(f"获取到数据: {len(data3)} 条")
    print(f"时间范围: {data3['datetime'].min()} 到 {data3['datetime'].max()}")
    
    print("\n=== 测试完成 ===")
    print("\n功能说明:")
    print("1. 默认情况下，获取从开始时间到现在的数据")
    print("2. 可以通过end_year和end_month参数指定结束时间")
    print("3. 支持灵活的时间范围配置")
    print("4. 修复了原有的数据获取错误")

if __name__ == "__main__":
    test_endtime_config()