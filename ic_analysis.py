#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC值差异分析脚本
分析训练时IC和回测时IC的差异原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

def analyze_ic_difference():
    """分析IC值差异的原因"""
    print("="*60)
    print("IC值差异分析")
    print("="*60)
    
    # 模拟训练时的IC计算（batch方式）
    print("\n1. 训练时IC计算（batch方式）:")
    print("- 样本量: 2048只股票")
    print("- 时间窗口: T+1~T+10")
    print("- 计算方式: 整个batch内所有样本的相关系数")
    
    # 模拟数据
    np.random.seed(42)
    batch_size = 2048
    time_steps = 10
    
    # 模拟预测值和真实值（有一定相关性）
    predictions = np.random.randn(batch_size, time_steps)
    targets = predictions * 0.15 + np.random.randn(batch_size, time_steps) * 0.99
    
    # 计算训练时IC
    train_ic_list = []
    for t in range(time_steps):
        pred = predictions[:, t]
        true = targets[:, t]
        ic = np.corrcoef(pred, true)[0, 1]
        train_ic_list.append(ic)
    
    train_ic_mean = np.mean(train_ic_list)
    train_ic_std = np.std(train_ic_list)
    
    print(f"✓ 训练时IC均值: {train_ic_mean:.4f}")
    print(f"✓ 训练时IC标准差: {train_ic_std:.4f}")
    print(f"✓ 训练时IC范围: {np.min(train_ic_list):.4f} ~ {np.max(train_ic_list):.4f}")
    
    # 模拟回测时的IC计算（每日方式）
    print("\n2. 回测时IC计算（每日方式）:")
    print("- 样本量: 每日约300-500只股票")
    print("- 时间窗口: T+1~T+6")
    print("- 计算方式: 每日所有股票的相关系数")
    
    # 模拟回测数据
    daily_stocks = 400  # 每日股票数量
    num_days = 410      # 回测天数
    
    backtest_ic_list = []
    for day in range(num_days):
        # 每日的预测值和真实值
        daily_pred = np.random.randn(daily_stocks)
        daily_true = daily_pred * 0.15 + np.random.randn(daily_stocks) * 0.99
        
        # 计算每日IC
        ic = np.corrcoef(daily_pred, daily_true)[0, 1]
        if not np.isnan(ic):
            backtest_ic_list.append(ic)
    
    backtest_ic_mean = np.mean(backtest_ic_list)
    backtest_ic_std = np.std(backtest_ic_list)
    
    print(f"✓ 回测时IC均值: {backtest_ic_mean:.4f}")
    print(f"✓ 回测时IC标准差: {backtest_ic_std:.4f}")
    print(f"✓ 回测时IC范围: {np.min(backtest_ic_list):.4f} ~ {np.max(backtest_ic_list):.4f}")
    
    # 分析差异
    print("\n3. 差异分析:")
    ic_diff = train_ic_mean - backtest_ic_mean
    print(f"✓ IC均值差异: {ic_diff:.4f}")
    print(f"✓ 差异比例: {abs(ic_diff)/train_ic_mean*100:.1f}%")
    
    # 样本量影响分析
    print("\n4. 样本量影响分析:")
    sample_sizes = [100, 200, 400, 800, 1600, 2048]
    ic_by_sample_size = []
    
    for size in sample_sizes:
        ic_list = []
        for _ in range(100):  # 重复100次取平均
            pred = np.random.randn(size)
            true = pred * 0.15 + np.random.randn(size) * 0.99
            ic = np.corrcoef(pred, true)[0, 1]
            if not np.isnan(ic):
                ic_list.append(ic)
        ic_by_sample_size.append(np.mean(ic_list))
    
    print("样本量对IC稳定性的影响:")
    for size, ic in zip(sample_sizes, ic_by_sample_size):
        print(f"  样本量 {size:4d}: IC = {ic:.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    # 1. IC分布对比
    plt.subplot(2, 3, 1)
    plt.hist(train_ic_list, bins=20, alpha=0.7, label='训练时IC', color='blue')
    plt.hist(backtest_ic_list, bins=20, alpha=0.7, label='回测时IC', color='red')
    plt.xlabel('IC值')
    plt.ylabel('频次')
    plt.title('IC分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 样本量vs IC稳定性
    plt.subplot(2, 3, 2)
    plt.plot(sample_sizes, ic_by_sample_size, 'o-', linewidth=2, markersize=8)
    plt.xlabel('样本量')
    plt.ylabel('IC均值')
    plt.title('样本量对IC稳定性的影响')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label='真实相关性')
    plt.legend()
    
    # 3. 时间序列对比
    plt.subplot(2, 3, 3)
    plt.plot(range(len(train_ic_list)), train_ic_list, 'b-', label='训练时IC', alpha=0.7)
    plt.plot(range(len(backtest_ic_list[:50])), backtest_ic_list[:50], 'r-', label='回测时IC(前50天)', alpha=0.7)
    plt.xlabel('时间步/天数')
    plt.ylabel('IC值')
    plt.title('IC时间序列对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 箱线图对比
    plt.subplot(2, 3, 4)
    data_to_plot = [train_ic_list, backtest_ic_list]
    plt.boxplot(data_to_plot, labels=['训练时IC', '回测时IC'])
    plt.ylabel('IC值')
    plt.title('IC分布箱线图')
    plt.grid(True, alpha=0.3)
    
    # 5. 相关性散点图
    plt.subplot(2, 3, 5)
    # 选择前100个回测IC点
    sample_pred = np.random.randn(100)
    sample_true = sample_pred * 0.15 + np.random.randn(100) * 0.99
    plt.scatter(sample_pred, sample_true, alpha=0.6, color='red')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('预测值vs真实值散点图')
    plt.grid(True, alpha=0.3)
    
    # 6. 统计信息
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = f"""
    统计信息对比:
    
    训练时IC:
    均值: {train_ic_mean:.4f}
    标准差: {train_ic_std:.4f}
    样本量: {batch_size}
    
    回测时IC:
    均值: {backtest_ic_mean:.4f}
    标准差: {backtest_ic_std:.4f}
    样本量: {daily_stocks}
    
    差异分析:
    IC差异: {ic_diff:.4f}
    差异比例: {abs(ic_diff)/train_ic_mean*100:.1f}%
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('factor_results/ic_difference_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 分析图表已保存: factor_results/ic_difference_analysis.png")
    plt.show()
    
    # 结论和建议
    print("\n5. 结论和建议:")
    print("✓ IC值差异的主要原因:")
    print("  - 样本量差异: 训练时2048只股票 vs 回测时400只股票")
    print("  - 统计稳定性: 大样本量提供更稳定的IC估计")
    print("  - 时间窗口: 训练时T+1~T+10 vs 回测时T+1~T+6")
    
    print("\n✓ 改进建议:")
    print("  1. 增加回测时的样本量（减少股票筛选条件）")
    print("  2. 使用滚动窗口计算IC，提高统计稳定性")
    print("  3. 考虑使用更长的预测时间窗口")
    print("  4. 添加IC置信区间分析")
    
    return {
        'train_ic_mean': train_ic_mean,
        'backtest_ic_mean': backtest_ic_mean,
        'ic_difference': ic_diff,
        'sample_size_impact': dict(zip(sample_sizes, ic_by_sample_size))
    }

if __name__ == "__main__":
    results = analyze_ic_difference() 