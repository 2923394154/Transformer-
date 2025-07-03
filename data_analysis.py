#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
等量K线数据分析和特征构建脚本
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def analyze_data_format():
    """分析数据文件格式"""
    print("=== 数据文件格式分析 ===")
    
    # 分析stock_price_vol_d.txt
    print("\n1. 股票价格成交量数据 (stock_price_vol_d.txt):")
    try:
        # 尝试读取前几行
        with open('stock_price_vol_d.txt', 'r', encoding='utf-8') as f:
            lines = [f.readline().strip() for _ in range(10)]
        
        print("前10行数据:")
        for i, line in enumerate(lines, 1):
            if line:
                print(f"行{i}: {line}")
        
        # 尝试判断分隔符和列数
        if lines[0]:
            separators = [',', '\t', ' ', '|', ';']
            for sep in separators:
                cols = lines[0].split(sep)
                if len(cols) > 1:
                    print(f"可能的分隔符: '{sep}', 列数: {len(cols)}")
                    print(f"列内容预览: {cols[:5]}")
                    break
    
    except Exception as e:
        print(f"读取股票数据文件出错: {e}")
    
    # 分析barra_Exposure(2).
    print("\n2. Barra暴露度数据 (barra_Exposure(2).):")
    try:
        with open('barra_Exposure(2).', 'r', encoding='utf-8') as f:
            lines = [f.readline().strip() for _ in range(10)]
        
        print("前10行数据:")
        for i, line in enumerate(lines, 1):
            if line:
                print(f"行{i}: {line}")
                
        # 尝试判断分隔符和列数
        if lines[0]:
            separators = [',', '\t', ' ', '|', ';']
            for sep in separators:
                cols = lines[0].split(sep)
                if len(cols) > 1:
                    print(f"可能的分隔符: '{sep}', 列数: {len(cols)}")
                    print(f"列内容预览: {cols[:5]}")
                    break
                    
    except Exception as e:
        print(f"读取Barra数据文件出错: {e}")

def load_stock_data(file_path='stock_price_vol_d.txt'):
    """加载股票数据"""
    try:
        # 先尝试常见的分隔符
        for sep in [',', '\t', ' ', '|']:
            try:
                df = pd.read_csv(file_path, sep=sep, nrows=5)
                if df.shape[1] > 1:
                    print(f"成功读取数据，使用分隔符: '{sep}'")
                    print(f"数据形状: {df.shape}")
                    print(f"列名: {list(df.columns)}")
                    print("前5行数据:")
                    print(df.head())
                    
                    # 读取完整数据
                    df_full = pd.read_csv(file_path, sep=sep)
                    return df_full
            except:
                continue
        
        print("无法自动识别分隔符，尝试手动解析...")
        return None
        
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None

if __name__ == "__main__":
    # 分析数据格式
    analyze_data_format()
    
    print("\n" + "="*50)
    print("尝试加载股票数据...")
    
    # 加载股票数据
    stock_data = load_stock_data()
    
    if stock_data is not None:
        print(f"\n股票数据加载成功!")
        print(f"数据形状: {stock_data.shape}")
        print(f"列名: {list(stock_data.columns)}")
        print(f"数据类型:")
        print(stock_data.dtypes)
    else:
        print("股票数据加载失败，需要进一步分析数据格式") 