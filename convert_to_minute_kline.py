import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def convert_to_minute_kline(input_file='future_taobao_ss2401.csv', output_file='minute_kline_data.csv'):
    """
    将高频交易数据转换为分钟级K线数据
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return False
    
    # 打印文件大小
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # 转换为MB
    print(f"输入文件大小: {file_size:.2f} MB")
    
    try:
        # 使用chunksize参数分批读取大文件
        print("正在读取数据文件...")
        chunk_size = 100000  # 每次读取10万行
        chunks = pd.read_csv(input_file, chunksize=chunk_size)
        
        # 创建一个空的DataFrame来存储所有分钟K线数据
        all_minute_bars = pd.DataFrame()
        
        # 处理每个数据块
        chunk_count = 0
        for chunk in chunks:
            chunk_count += 1
            print(f"正在处理第 {chunk_count} 个数据块...")
            
            # 确保datetime列是正确的日期时间格式
            chunk['datetime'] = pd.to_datetime(chunk['datetime'])
            
            # 提取分钟级时间戳（去掉秒和微秒）
            chunk['minute'] = chunk['datetime'].dt.floor('min')
            
            # 按分钟分组，计算每分钟的开盘价和收盘价
            minute_bars = chunk.groupby('minute').agg(
                Open=('last_price', 'first'),
                Close=('last_price', 'last'),
                Volume=('volume', 'sum')
            ).reset_index()
            
            # 重命名列以符合要求
            minute_bars = minute_bars.rename(columns={'minute': 'DateTime'})
            
            # 添加到总结果中
            all_minute_bars = pd.concat([all_minute_bars, minute_bars])
        
        # 对合并后的结果按时间排序并去重（可能有重叠的分钟数据）
        print("正在整合所有数据块的结果...")
        all_minute_bars = all_minute_bars.sort_values('DateTime').drop_duplicates(subset=['DateTime'])
        
        # 保存结果到新的CSV文件
        print(f"正在保存结果到 {output_file}...")
        all_minute_bars.to_csv(output_file, index=True)
        
        print(f"处理完成！结果已保存到 {output_file}")
        print(f"转换后的数据共有 {len(all_minute_bars)} 条记录")
        
        # 显示前10条数据作为示例
        print("\n转换后的数据示例:")
        print(all_minute_bars.head(10))
        
        return True
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# 执行主程序
if __name__ == "__main__":
    convert_to_minute_kline()