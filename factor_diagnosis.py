#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer因子诊断脚本
分析为什么结果与期望的人工特征集差距很大
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def diagnose_factor_performance():
    """诊断因子性能问题"""
    print("="*80)
    print("Transformer因子性能诊断")
    print("="*80)
    
    # 1. 检查因子得分数据
    print("\n1. 检查因子得分数据...")
    factor_file = 'factor_results/ensemble_factor_scores.csv'
    if os.path.exists(factor_file):
        factor_scores = pd.read_csv(factor_file)
        factor_scores['date'] = pd.to_datetime(factor_scores['date'])
        print(f"✓ 因子得分数据: {factor_scores.shape}")
        print(f"✓ 时间范围: {factor_scores['date'].min()} ~ {factor_scores['date'].max()}")
        print(f"✓ 股票数量: {factor_scores['StockID'].nunique()}")
        print(f"✓ 因子得分统计:")
        print(f"  - 均值: {factor_scores['factor_score'].mean():.6f}")
        print(f"  - 标准差: {factor_scores['factor_score'].std():.6f}")
        print(f"  - 最小值: {factor_scores['factor_score'].min():.6f}")
        print(f"  - 最大值: {factor_scores['factor_score'].max():.6f}")
        
        # 检查每个日期的股票数量
        daily_stock_counts = factor_scores.groupby('date')['StockID'].count()
        print(f"✓ 每日股票数量统计:")
        print(f"  - 平均: {daily_stock_counts.mean():.1f}")
        print(f"  - 中位数: {daily_stock_counts.median():.1f}")
        print(f"  - 最小值: {daily_stock_counts.min()}")
        print(f"  - 最大值: {daily_stock_counts.max()}")
        
        # 检查股票数量不足的日期
        insufficient_dates = daily_stock_counts[daily_stock_counts < 50]
        print(f"✓ 股票数量不足50只的日期: {len(insufficient_dates)} 个")
        if len(insufficient_dates) > 0:
            print(f"  - 占比: {len(insufficient_dates)/len(daily_stock_counts)*100:.1f}%")
    else:
        print("✗ 因子得分文件不存在")
        return
    
    # 2. 检查股票价格数据
    print("\n2. 检查股票价格数据...")
    try:
        stock_data = pd.read_feather('stock_price_vol_d.txt')
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        print(f"✓ 股票价格数据: {stock_data.shape}")
        print(f"✓ 时间范围: {stock_data['date'].min()} ~ {stock_data['date'].max()}")
        print(f"✓ 股票数量: {stock_data['StockID'].nunique()}")
        
        # 检查每日股票数量
        daily_stock_counts = stock_data.groupby('date')['StockID'].count()
        print(f"✓ 每日股票数量统计:")
        print(f"  - 平均: {daily_stock_counts.mean():.1f}")
        print(f"  - 中位数: {daily_stock_counts.median():.1f}")
        print(f"  - 最小值: {daily_stock_counts.min()}")
        print(f"  - 最大值: {daily_stock_counts.max()}")
    except Exception as e:
        print(f"✗ 加载股票价格数据失败: {e}")
    
    # 3. 检查数据匹配情况
    print("\n3. 检查数据匹配情况...")
    common_dates = set(factor_scores['date'].unique()) & set(stock_data['date'].unique())
    common_stocks = set(factor_scores['StockID'].unique()) & set(stock_data['StockID'].unique())
    
    print(f"✓ 共同日期数量: {len(common_dates)}")
    print(f"✓ 共同股票数量: {len(common_stocks)}")
    print(f"✓ 因子数据日期覆盖率: {len(common_dates)/len(factor_scores['date'].unique())*100:.1f}%")
    print(f"✓ 因子数据股票覆盖率: {len(common_stocks)/len(factor_scores['StockID'].unique())*100:.1f}%")
    
    # 4. 检查因子得分分布
    print("\n4. 检查因子得分分布...")
    recent_factors = factor_scores[factor_scores['date'] >= '2024-01-01']
    if len(recent_factors) > 0:
        print(f"✓ 2024年后的因子得分统计:")
        print(f"  - 均值: {recent_factors['factor_score'].mean():.6f}")
        print(f"  - 标准差: {recent_factors['factor_score'].std():.6f}")
        print(f"  - 最小值: {recent_factors['factor_score'].min():.6f}")
        print(f"  - 最大值: {recent_factors['factor_score'].max():.6f}")
        
        # 检查因子得分是否过于集中
        score_std = recent_factors['factor_score'].std()
        if score_std < 0.01:
            print("⚠️  警告: 因子得分标准差过小，可能缺乏区分度")
        elif score_std > 1.0:
            print("⚠️  警告: 因子得分标准差过大，可能存在异常值")
        else:
            print("✓ 因子得分标准差正常")
    
    # 5. 检查训练模型信息
    print("\n5. 检查训练模型信息...")
    ensemble_info_file = 'trained_models/ensemble_info.json'
    if os.path.exists(ensemble_info_file):
        import json
        with open(ensemble_info_file, 'r') as f:
            ensemble_info = json.load(f)
        
        print(f"✓ 集成模型信息:")
        print(f"  - 模型数量: {ensemble_info['num_models']}")
        print(f"  - 种子列表: {ensemble_info['seeds']}")
        
        for model in ensemble_info['models']:
            print(f"  - Seed {model['seed']}: IC = {model['best_ic']:.6f}")
        
        # 检查训练配置
        config = ensemble_info['models'][0]['config']
        print(f"✓ 训练配置:")
        print(f"  - 最大股票数: 未指定（可能使用了所有股票）")
        print(f"  - 批大小: {config['training_params']['batch_size']}")
        print(f"  - 学习率: {config['training_params']['learning_rate']}")
        print(f"  - 训练轮数: {config['training_params']['num_epochs']}")
    
    # 6. 问题诊断和建议
    print("\n" + "="*80)
    print("问题诊断和建议")
    print("="*80)
    
    print("\n主要问题:")
    print("1. 股票池覆盖不足:")
    print("   - 训练时可能只用了少量股票")
    print("   - 回测时要求50只以上股票，但因子数据覆盖不足")
    
    print("\n2. 时间窗口不匹配:")
    print("   - 训练时预测T+1~T+10的收益")
    print("   - 回测时只用T+1的收益")
    print("   - 建议: 使用T+1的收益进行回测")
    
    print("\n3. 数据预处理不一致:")
    print("   - 训练和回测的标准化参数可能不同")
    print("   - 建议: 确保使用相同的预处理参数")
    
    print("\n改进建议:")
    print("1. 重新训练模型，使用更多股票（如100-200只）")
    print("2. 修改回测逻辑，使用T+1的收益而不是T+1~T+10")
    print("3. 检查因子得分的标准化处理")
    print("4. 增加股票池筛选的灵活性")
    
    print("\n" + "="*80)

def check_factor_vs_returns():
    """检查因子得分与收益率的匹配情况"""
    print("\n检查因子得分与收益率匹配情况...")
    
    # 加载数据
    factor_scores = pd.read_csv('factor_results/ensemble_factor_scores.csv')
    factor_scores['date'] = pd.to_datetime(factor_scores['date'])
    
    stock_data = pd.read_feather('stock_price_vol_d.txt')
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    
    # 选择一个测试日期
    test_date = factor_scores['date'].max()
    print(f"测试日期: {test_date}")
    
    # 获取该日期的因子得分
    test_factors = factor_scores[factor_scores['date'] == test_date]
    print(f"因子得分股票数量: {len(test_factors)}")
    
    # 获取下一交易日的收益率
    next_date = stock_data[stock_data['date'] > test_date]['date'].min()
    if next_date is not None:
        print(f"下一交易日: {next_date}")
        
        # 计算收益率
        returns = []
        for _, row in test_factors.iterrows():
            stock_id = row['StockID']
            
            # 获取当前价格和下一日价格
            current_price = stock_data[
                (stock_data['date'] == test_date) & 
                (stock_data['StockID'] == stock_id)
            ]['close'].iloc[0] if len(stock_data[
                (stock_data['date'] == test_date) & 
                (stock_data['StockID'] == stock_id)
            ]) > 0 else None
            
            next_price = stock_data[
                (stock_data['date'] == next_date) & 
                (stock_data['StockID'] == stock_id)
            ]['close'].iloc[0] if len(stock_data[
                (stock_data['date'] == next_date) & 
                (stock_data['StockID'] == stock_id)
            ]) > 0 else None
            
            if current_price is not None and next_price is not None:
                stock_return = (next_price - current_price) / current_price
                returns.append({
                    'StockID': stock_id,
                    'factor_score': row['factor_score'],
                    'return': stock_return
                })
        
        if returns:
            returns_df = pd.DataFrame(returns)
            print(f"匹配的股票数量: {len(returns_df)}")
            
            # 计算IC和RankIC
            factor_values = returns_df['factor_score'].values
            return_values = returns_df['return'].values
            
            ic = np.corrcoef(factor_values, return_values)[0, 1]
            factor_ranks = rankdata(factor_values)
            return_ranks = rankdata(return_values)
            rank_ic = np.corrcoef(factor_ranks, return_ranks)[0, 1]
            
            print(f"单日IC: {ic:.6f}")
            print(f"单日RankIC: {rank_ic:.6f}")
            
            # 检查因子得分分布
            print(f"因子得分统计:")
            print(f"  - 均值: {factor_values.mean():.6f}")
            print(f"  - 标准差: {factor_values.std():.6f}")
            print(f"  - 范围: {factor_values.min():.6f} ~ {factor_values.max():.6f}")
            
            print(f"收益率统计:")
            print(f"  - 均值: {return_values.mean():.6f}")
            print(f"  - 标准差: {return_values.std():.6f}")
            print(f"  - 范围: {return_values.min():.6f} ~ {return_values.max():.6f}")

if __name__ == "__main__":
    diagnose_factor_performance()
    check_factor_vs_returns() 