#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间稳定性分析：解决样本RankIC很好但整体RankICIR很差的矛盾
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def analyze_temporal_stability():
    """分析因子的时间稳定性"""
    
    print("="*80)
    print("因子时间稳定性分析")
    print("="*80)
    print("目标：解决样本RankIC=0.233 vs 整体RankICIR=0.044的矛盾")
    print("="*80)
    
    # 1. 加载数据
    print("\n📊 加载完整数据集")
    print("-"*50)
    
    try:
        factor_scores = pd.read_csv('factor_results/ensemble_factor_scores.csv')
        factor_scores['date'] = pd.to_datetime(factor_scores['date'])
        
        try:
            stock_data = pd.read_feather('stock_price_vol_d.txt')
        except:
            stock_data = pd.read_csv('stock_price_vol_d.txt')
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        print(f"✓ 因子得分: {factor_scores.shape}")
        print(f"✓ 股票价格: {stock_data.shape}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 时间分段分析
    print("\n📅 时间分段分析")
    print("-"*50)
    
    # 按年份分组分析
    factor_scores['year'] = factor_scores['date'].dt.year
    years = sorted(factor_scores['year'].unique())
    
    print(f"数据覆盖年份: {years}")
    
    yearly_results = []
    
    for year in years:
        print(f"\n分析 {year} 年数据...")
        
        year_factors = factor_scores[factor_scores['year'] == year]
        
        if len(year_factors) < 50:  # 数据太少则跳过
            print(f"  {year}年数据太少，跳过")
            continue
        
        # 计算该年的收益率（使用样本）
        sample_size = min(200, len(year_factors))
        year_sample = year_factors.sample(n=sample_size, random_state=42)
        
        returns_list = []
        
        for _, factor_row in year_sample.iterrows():
            stock_id = factor_row['StockID']
            factor_date = factor_row['date']
            factor_score = factor_row['factor_score']
            
            # 获取价格数据
            stock_prices = stock_data[stock_data['StockID'] == stock_id].sort_values('date')
            
            current_price_data = stock_prices[stock_prices['date'] == factor_date]
            if len(current_price_data) == 0:
                continue
            
            start_price = current_price_data.iloc[0]['close']
            
            # 计算10日后收益率
            future_prices = stock_prices[stock_prices['date'] > factor_date].head(10)
            if len(future_prices) >= 10:
                end_price = future_prices.iloc[-1]['close']
                forward_return = (end_price - start_price) / start_price
                
                returns_list.append({
                    'factor_score': factor_score,
                    'forward_return': forward_return
                })
            
            # 限制计算量
            if len(returns_list) >= 50:
                break
        
        if len(returns_list) >= 20:  # 至少需要20个样本
            returns_df = pd.DataFrame(returns_list)
            
            try:
                rank_ic, p_value = spearmanr(
                    returns_df['factor_score'], 
                    returns_df['forward_return']
                )
                
                yearly_results.append({
                    'year': year,
                    'rank_ic': rank_ic,
                    'p_value': p_value,
                    'sample_size': len(returns_list),
                    'factor_count': len(year_factors)
                })
                
                print(f"  {year}年 RankIC: {rank_ic:.4f} (p={p_value:.3f}, n={len(returns_list)})")
                
            except Exception as e:
                print(f"  {year}年 计算失败: {e}")
        else:
            print(f"  {year}年 样本不足")
    
    # 3. 结果汇总分析
    print("\n📊 时间稳定性汇总")
    print("-"*50)
    
    if len(yearly_results) > 0:
        results_df = pd.DataFrame(yearly_results)
        
        print("各年份RankIC表现:")
        for _, row in results_df.iterrows():
            status = "✓" if abs(row['rank_ic']) > 0.05 else "❌"
            print(f"  {row['year']}: {status} RankIC={row['rank_ic']:7.4f} (p={row['p_value']:.3f})")
        
        # 统计分析
        avg_rank_ic = results_df['rank_ic'].mean()
        std_rank_ic = results_df['rank_ic'].std()
        positive_years = (results_df['rank_ic'] > 0).sum()
        significant_years = (results_df['p_value'] < 0.05).sum()
        
        print(f"\n时间稳定性统计:")
        print(f"  平均RankIC: {avg_rank_ic:.4f}")
        print(f"  标准差: {std_rank_ic:.4f}")
        print(f"  正向年份: {positive_years}/{len(results_df)} ({positive_years/len(results_df)*100:.1f}%)")
        print(f"  显著年份: {significant_years}/{len(results_df)} ({significant_years/len(results_df)*100:.1f}%)")
        
        # 4. 问题诊断
        print("\n🔍 问题诊断")
        print("-"*50)
        
        if std_rank_ic > 0.1:
            print("❌ 发现严重的时间不稳定性")
            print("  因子表现在不同年份差异巨大")
            print("  这解释了为什么样本表现好但整体表现差")
        
        if positive_years < len(results_df) * 0.6:
            print("❌ 因子负向表现年份过多")
            print("  超过40%的年份表现为负，影响整体表现")
        
        if significant_years < len(results_df) * 0.5:
            print("❌ 统计显著性不稳定")
            print("  超过50%的年份不具有统计显著性")
        
        # 寻找最好和最差的年份
        best_year = results_df.loc[results_df['rank_ic'].idxmax()]
        worst_year = results_df.loc[results_df['rank_ic'].idxmin()]
        
        print(f"\n表现对比:")
        print(f"  最佳年份: {best_year['year']} (RankIC={best_year['rank_ic']:.4f})")
        print(f"  最差年份: {worst_year['year']} (RankIC={worst_year['rank_ic']:.4f})")
        print(f"  差异: {best_year['rank_ic'] - worst_year['rank_ic']:.4f}")
        
    else:
        print("❌ 无法完成时间稳定性分析")
    
    # 5. 改进建议
    print("\n💡 基于时间稳定性的改进建议")
    print("-"*50)
    
    if len(yearly_results) > 0:
        if std_rank_ic > 0.1:
            print("针对时间不稳定性问题:")
            print("  1. 增加时间特征 - 考虑市场周期、季节性因素")
            print("  2. 滚动训练 - 使用滑动窗口重新训练模型")
            print("  3. 市场制度检测 - 识别不同市场环境并分别建模")
            print("  4. 正则化增强 - 提高模型的时间泛化能力")
        
        if positive_years < len(results_df) * 0.6:
            print("针对负向年份过多:")
            print("  1. 特征工程 - 添加更稳定的因子")
            print("  2. 集成学习 - 结合多种不同类型的模型")
            print("  3. 动态调权 - 根据市场环境调整因子权重")
    
    print("\n" + "="*80)
    print("时间稳定性分析完成")
    if len(yearly_results) > 0:
        print(f"核心发现: 时间稳定性标准差 = {std_rank_ic:.4f}")
        if std_rank_ic > 0.1:
            print("结论: 因子存在严重的时间不稳定性，这是RankICIR偏低的主要原因")
        else:
            print("结论: 时间稳定性较好，问题可能在其他方面")
    print("="*80)

if __name__ == "__main__":
    analyze_temporal_stability() 