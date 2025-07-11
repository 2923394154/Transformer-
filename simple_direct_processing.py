#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化直接股票数据处理脚本 - GPU加速版本
直接处理原始K线数据，输出格式匹配现有训练脚本
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# GPU加速相关导入
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
    print("✓ GPU加速可用 (CuPy + cuDF)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ GPU加速不可用，使用CPU处理")

def process_stock_data():
    """直接处理股票数据，生成训练特征 - GPU加速版本"""
    print("简化直接股票数据处理 - GPU加速版本")
    print("="*50)
    
    if GPU_AVAILABLE:
        print("使用GPU加速处理")
    else:
        print("使用CPU处理")
    
    # 生成新的文件名，不覆盖原有文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"direct_features_{timestamp}.csv"
    print(f"将生成新文件: {output_file}")
    
    # 尝试读取原始数据 - 应用与原有文件一致的feather读取方法
    file_path = 'stock_price_vol_d.txt'
    stock_data = None
    
    if os.path.exists(file_path):
        print(f"发现数据文件: {file_path}")
        try:
            # 首选：尝试读取feather格式数据
            print("  尝试读取feather格式...")
            stock_data = pd.read_feather(file_path)
            print(f"  ✓ 成功以feather格式加载数据: {stock_data.shape}")
        except Exception as e:
            print(f"  feather格式读取失败: {e}")
            try:
                # 备选：尝试读取CSV格式
                print("  尝试读取CSV格式...")
                stock_data = pd.read_csv(file_path)
                print(f"  ✓ 成功以CSV格式加载数据: {stock_data.shape}")
            except Exception as e2:
                print(f"  CSV格式读取失败: {e2}")
                try:
                    # 最后尝试：手动解析文本格式
                    print("  尝试手动解析文本格式...")
                    stock_data = read_text_data(file_path)
                    if stock_data is not None:
                        print(f"  ✓ 成功以文本格式加载数据: {stock_data.shape}")
                except Exception as e3:
                    print(f"  文本格式解析失败: {e3}")
    else:
        print(f"✗ 数据文件不存在: {file_path}")
        return False
    
    if stock_data is None:
        print("✗ 无法加载原始数据")
        return False
    
    # 数据预处理 - 与原有文件保持一致
    print("正在预处理数据...")
    
    # 标准化列名
    if 'volume' in stock_data.columns:
        stock_data = stock_data.rename(columns={'volume': 'vol'})
    
    # 数据类型转换 - 与原有文件一致
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    numeric_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
    for col in numeric_columns:
        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
    
    # 删除缺失值
    stock_data = stock_data.dropna(subset=numeric_columns)
    stock_data = stock_data.sort_values(['StockID', 'date'])
    
    # 筛选时间范围：2013年至今
    start_date = pd.to_datetime('2013-01-01')
    stock_data = stock_data[stock_data['date'] >= start_date]
    
    print(f"✓ 预处理完成: {stock_data.shape}")
    print(f"✓ 数据时间范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
    print(f"✓ 股票数量: {stock_data['StockID'].nunique()}")
    
    # 筛选有效股票
    stock_counts = stock_data.groupby('StockID').size()
    valid_stocks = list(stock_counts[stock_counts >= 50].index)  # 至少50天数据
    
    print(f"✓ 有效股票: {len(valid_stocks)} 只")
    
    # 生成特征数据 - GPU加速版本
    print("正在生成特征数据...")
    feature_records = []
    
    if GPU_AVAILABLE:
        # GPU加速处理
        feature_records = process_features_gpu(stock_data, valid_stocks)
    else:
        # CPU处理
        feature_records = process_features_cpu(stock_data, valid_stocks)
    
    # 创建DataFrame
    features_df = pd.DataFrame(feature_records)
    features_df = features_df.sort_values(['end_date', 'StockID']).reset_index(drop=True)
    
    # 保存到新文件
    features_df.to_csv(output_file, index=False)
    
    print(f"✓ 特征数据已保存: {output_file}")
    print(f"✓ 记录数: {len(features_df)} 条")
    print(f"✓ 股票数量: {features_df['StockID'].nunique()}")
    print(f"✓ 时间范围: {features_df['end_date'].min()} ~ {features_df['end_date'].max()}")
    
    # 创建符号链接，方便训练脚本使用
    link_file = "direct_features_latest.csv"
    if os.path.exists(link_file):
        os.remove(link_file)
    os.symlink(output_file, link_file)
    print(f"✓ 创建符号链接: {link_file} -> {output_file}")
    
    return output_file

def process_features_cpu(stock_data, valid_stocks):
    """CPU处理特征生成"""
    print("  使用CPU处理...")
    feature_records = []
    
    for stock_id in tqdm(valid_stocks, desc="处理股票"):
        stock_df = stock_data[stock_data['StockID'] == stock_id].copy()
        
        # 按40天窗口提取特征
        for i in range(40, len(stock_df), 5):  # 每5天采样一次
            window = stock_df.iloc[i-40:i]
            
            if len(window) == 40:
                # 取最后一天的特征
                last_day = window.iloc[-1]
                
                record = {
                    'StockID': stock_id,
                    'end_date': last_day['date'],
                    'open': last_day['open'],
                    'high': last_day['high'],
                    'low': last_day['low'],
                    'close': last_day['close'],
                    'vwap': last_day['close'],  # 简化VWAP
                    'amount': last_day['amount']
                }
                
                feature_records.append(record)
    
    return feature_records

def process_features_gpu(stock_data, valid_stocks):
    """GPU加速处理特征生成"""
    print("  使用GPU加速处理...")
    
    # 转换为cuDF DataFrame
    try:
        gpu_df = cudf.from_pandas(stock_data)
        print(f"  ✓ GPU数据加载完成: {gpu_df.shape}")
    except Exception as e:
        print(f"  ⚠ GPU转换失败，回退到CPU: {e}")
        return process_features_cpu(stock_data, valid_stocks)
    
    feature_records = []
    
    # 批量处理所有股票
    for stock_id in tqdm(valid_stocks, desc="GPU处理股票"):
        try:
            # GPU筛选股票数据
            stock_mask = gpu_df['StockID'] == stock_id
            stock_gpu = gpu_df[stock_mask].copy()
            
            if len(stock_gpu) < 40:
                continue
            
            # 转换为numpy进行窗口处理
            stock_np = stock_gpu.to_pandas()
            
            # 按40天窗口提取特征
            for i in range(40, len(stock_np), 5):  # 每5天采样一次
                window = stock_np.iloc[i-40:i]
                
                if len(window) == 40:
                    # 取最后一天的特征
                    last_day = window.iloc[-1]
                    
                    record = {
                        'StockID': stock_id,
                        'end_date': last_day['date'],
                        'open': last_day['open'],
                        'high': last_day['high'],
                        'low': last_day['low'],
                        'close': last_day['close'],
                        'vwap': last_day['close'],  # 简化VWAP
                        'amount': last_day['amount']
                    }
                    
                    feature_records.append(record)
                    
        except Exception as e:
            print(f"  ⚠ GPU处理股票 {stock_id} 失败: {e}")
            continue
    
    return feature_records

def read_text_data(file_path):
    """读取文本格式数据"""
    try:
        with open(file_path, 'r', encoding='latin1') as f:
            lines = f.readlines()
        
        data = []
        for line_num, line in enumerate(lines[1:], 2):  # 从第2行开始
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                try:
                    # 检查空字符串并跳过
                    if any(part.strip() == '' for part in parts[2:7]):
                        continue
                    
                    data.append({
                        'StockID': parts[0],
                        'date': parts[1],
                        'open': float(parts[2]) if parts[2].strip() else 0.0,
                        'high': float(parts[3]) if parts[3].strip() else 0.0,
                        'low': float(parts[4]) if parts[4].strip() else 0.0,
                        'close': float(parts[5]) if parts[5].strip() else 0.0,
                        'vol': float(parts[6]) if parts[6].strip() else 0.0,
                        'amount': float(parts[7]) if len(parts) > 7 and parts[7].strip() else 0.0
                    })
                except (ValueError, IndexError) as e:
                    print(f"跳过第{line_num}行数据错误: {e}")
                    continue
        
        if not data:
            print("没有有效数据行")
            return None
            
        df = pd.DataFrame(data)
        print(f"成功读取 {len(df)} 行数据")
        return df
        
    except Exception as e:
        print(f"文本数据读取失败: {e}")
        return None

if __name__ == "__main__":
    result = process_stock_data()
    if result:
        print(f"\n✓ 处理完成!")
        print(f"✓ 新数据文件: {result}")
        print(f"✓ 符号链接: direct_features_latest.csv")
        print(f"✓ 可直接用于训练脚本")
    else:
        print("\n✗ 处理失败") 