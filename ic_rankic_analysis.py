#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IC和RankIC差异分析脚本
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ICvsRankICAnalyzer:
    def __init__(self):
        print("IC和RankIC差异分析器初始化完成")
    
    def load_ic_data(self, ic_file='factor_results/enhanced_ic_data.csv'):
        """加载IC数据"""
        print("="*60)
        print("加载IC数据")
        print("="*60)
        
        if not os.path.exists(ic_file):
            print(f"✗ IC数据文件不存在: {ic_file}")
            return None
        
        print(f"正在加载IC数据: {ic_file}")
        ic_df = pd.read_csv(ic_file)
        ic_df['date'] = pd.to_datetime(ic_df['date'])
        
        print(f"✓ IC数据: {ic_df.shape}")
        print(f"✓ 时间范围: {ic_df['date'].min()} ~ {ic_df['date'].max()}")
        print(f"✓ 列名: {list(ic_df.columns)}")
        
        return ic_df
    
    def analyze_ic_rankic_difference(self, ic_df):
        """分析IC和RankIC差异的原因"""
        print("="*60)
        print("分析IC和RankIC差异原因")
        print("="*60)
        
        if len(ic_df) == 0:
            print("没有IC数据可供分析")
            return
        
        # 1. 基本统计信息
        print("基本统计信息:")
        print(f"  总日期数: {len(ic_df)}")
        print(f"  IC_5d均值: {ic_df['ic_5d'].mean():.6f}")
        print(f"  IC_5d标准差: {ic_df['ic_5d'].std():.6f}")
        print(f"  RankIC_5d均值: {ic_df['rank_ic_5d'].mean():.6f}")
        print(f"  RankIC_5d标准差: {ic_df['rank_ic_5d'].std():.6f}")
        
        # 2. 分析IC和RankIC的相关性
        ic_rank_corr = np.corrcoef(ic_df['ic_5d'], ic_df['rank_ic_5d'])[0, 1]
        print(f"\nIC和RankIC相关性: {ic_rank_corr:.6f}")
        
        # 3. 分析差异的分布
        ic_df['ic_rank_diff'] = ic_df['ic_5d'] - ic_df['rank_ic_5d']
        ic_df['ic_rank_diff_abs'] = abs(ic_df['ic_rank_diff'])
        
        print(f"\nIC-RankIC差异统计:")
        print(f"  均值: {ic_df['ic_rank_diff'].mean():.6f}")
        print(f"  标准差: {ic_df['ic_rank_diff'].std():.6f}")
        print(f"  最小值: {ic_df['ic_rank_diff'].min():.6f}")
        print(f"  最大值: {ic_df['ic_rank_diff'].max():.6f}")
        print(f"  绝对差异均值: {ic_df['ic_rank_diff_abs'].mean():.6f}")
        print(f"  绝对差异中位数: {ic_df['ic_rank_diff_abs'].median():.6f}")
        
        # 4. 找出差异最大的日期
        max_diff_idx = ic_df['ic_rank_diff_abs'].idxmax()
        max_diff_date = ic_df.loc[max_diff_idx, 'date']
        max_diff_ic = ic_df.loc[max_diff_idx, 'ic_5d']
        max_diff_rank_ic = ic_df.loc[max_diff_idx, 'rank_ic_5d']
        
        print(f"\n差异最大的日期: {max_diff_date}")
        print(f"  IC: {max_diff_ic:.6f}")
        print(f"  RankIC: {max_diff_rank_ic:.6f}")
        print(f"  差异: {abs(max_diff_ic - max_diff_rank_ic):.6f}")
        
        # 5. 分析差异分布
        print(f"\n差异分布分析:")
        print(f"  差异 > 0.05 的日期数: {(ic_df['ic_rank_diff_abs'] > 0.05).sum()}")
        print(f"  差异 > 0.1 的日期数: {(ic_df['ic_rank_diff_abs'] > 0.1).sum()}")
        print(f"  差异 > 0.2 的日期数: {(ic_df['ic_rank_diff_abs'] > 0.2).sum()}")
        
        # 6. 分析IC和RankIC的分布特征
        print(f"\n分布特征分析:")
        print(f"  IC偏度: {ic_df['ic_5d'].skew():.4f}")
        print(f"  IC峰度: {ic_df['ic_5d'].kurtosis():.4f}")
        print(f"  RankIC偏度: {ic_df['rank_ic_5d'].skew():.4f}")
        print(f"  RankIC峰度: {ic_df['rank_ic_5d'].kurtosis():.4f}")
        
        # 7. 分析不同时间段的表现
        ic_df['year'] = ic_df['date'].dt.year
        yearly_stats = ic_df.groupby('year').agg({
            'ic_5d': ['mean', 'std'],
            'rank_ic_5d': ['mean', 'std'],
            'ic_rank_diff_abs': 'mean'
        }).round(6)
        
        print(f"\n年度统计:")
        print(yearly_stats)
        
        # 8. 生成可能的原因分析
        possible_causes = []
        
        if abs(ic_rank_corr) < 0.8:
            possible_causes.append("IC和RankIC相关性较低，可能存在非线性关系")
        
        if ic_df['ic_5d'].skew() > 1.0 or ic_df['ic_5d'].skew() < -1.0:
            possible_causes.append("IC分布严重偏斜")
        
        if ic_df['ic_5d'].kurtosis() > 3.0:
            possible_causes.append("IC分布存在尖峰厚尾特征")
        
        if ic_df['ic_rank_diff_abs'].mean() > 0.05:
            possible_causes.append("IC和RankIC差异较大，可能存在异常值影响")
        
        # 检查是否有极端值
        ic_q99 = ic_df['ic_5d'].quantile(0.99)
        ic_q01 = ic_df['ic_5d'].quantile(0.01)
        rankic_q99 = ic_df['rank_ic_5d'].quantile(0.99)
        rankic_q01 = ic_df['rank_ic_5d'].quantile(0.01)
        
        if abs(ic_q99) > 0.3 or abs(ic_q01) > 0.3:
            possible_causes.append("IC存在极端值")
        
        if abs(rankic_q99) > 0.3 or abs(rankic_q01) > 0.3:
            possible_causes.append("RankIC存在极端值")
        
        print(f"\n可能的原因:")
        if possible_causes:
            for i, cause in enumerate(possible_causes, 1):
                print(f"  {i}. {cause}")
        else:
            print("  未发现明显的问题")
        
        # 9. 生成可视化
        self.generate_visualizations(ic_df)
        
        return ic_df
    
    def generate_visualizations(self, ic_df):
        """生成可视化图表"""
        print("="*60)
        print("生成可视化图表")
        print("="*60)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('IC和RankIC差异分析', fontsize=16, fontweight='bold')
        
        # 1. IC vs RankIC 散点图
        axes[0, 0].scatter(ic_df['ic_5d'], ic_df['rank_ic_5d'], alpha=0.6, s=20)
        axes[0, 0].plot([-0.3, 0.3], [-0.3, 0.3], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('IC_5d')
        axes[0, 0].set_ylabel('RankIC_5d')
        axes[0, 0].set_title('IC vs RankIC 散点图')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. IC和RankIC时间序列
        axes[0, 1].plot(ic_df['date'], ic_df['ic_5d'], label='IC_5d', alpha=0.7)
        axes[0, 1].plot(ic_df['date'], ic_df['rank_ic_5d'], label='RankIC_5d', alpha=0.7)
        axes[0, 1].set_xlabel('日期')
        axes[0, 1].set_ylabel('IC值')
        axes[0, 1].set_title('IC和RankIC时间序列')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. IC-RankIC差异时间序列
        axes[0, 2].plot(ic_df['date'], ic_df['ic_rank_diff'], alpha=0.7)
        axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 2].set_xlabel('日期')
        axes[0, 2].set_ylabel('IC - RankIC')
        axes[0, 2].set_title('IC-RankIC差异时间序列')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. IC分布直方图
        axes[1, 0].hist(ic_df['ic_5d'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('IC_5d')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('IC分布直方图')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. RankIC分布直方图
        axes[1, 1].hist(ic_df['rank_ic_5d'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('RankIC_5d')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('RankIC分布直方图')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 差异分布直方图
        axes[1, 2].hist(ic_df['ic_rank_diff'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('IC - RankIC')
        axes[1, 2].set_ylabel('频数')
        axes[1, 2].set_title('IC-RankIC差异分布')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('factor_results', exist_ok=True)
        plt.savefig('factor_results/ic_rankic_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图表已保存: factor_results/ic_rankic_analysis.png")
        
        # 显示图表
        plt.show()
    
    def generate_detailed_report(self, ic_df):
        """生成详细分析报告"""
        print("="*60)
        print("生成详细分析报告")
        print("="*60)
        
        # 计算统计指标
        ic_mean = ic_df['ic_5d'].mean()
        ic_std = ic_df['ic_5d'].std()
        rankic_mean = ic_df['rank_ic_5d'].mean()
        rankic_std = ic_df['rank_ic_5d'].std()
        ic_rank_corr = np.corrcoef(ic_df['ic_5d'], ic_df['rank_ic_5d'])[0, 1]
        avg_diff = ic_df['ic_rank_diff_abs'].mean()
        
        # 生成报告
        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_dates': len(ic_df),
            'time_range': f"{ic_df['date'].min().strftime('%Y-%m-%d')} ~ {ic_df['date'].max().strftime('%Y-%m-%d')}",
            'ic_statistics': {
                'mean': ic_mean,
                'std': ic_std,
                'skewness': ic_df['ic_5d'].skew(),
                'kurtosis': ic_df['ic_5d'].kurtosis(),
                'min': ic_df['ic_5d'].min(),
                'max': ic_df['ic_5d'].max()
            },
            'rankic_statistics': {
                'mean': rankic_mean,
                'std': rankic_std,
                'skewness': ic_df['rank_ic_5d'].skew(),
                'kurtosis': ic_df['rank_ic_5d'].kurtosis(),
                'min': ic_df['rank_ic_5d'].min(),
                'max': ic_df['rank_ic_5d'].max()
            },
            'difference_analysis': {
                'correlation': ic_rank_corr,
                'avg_absolute_diff': avg_diff,
                'max_absolute_diff': ic_df['ic_rank_diff_abs'].max(),
                'dates_with_large_diff': (ic_df['ic_rank_diff_abs'] > 0.1).sum()
            },
            'conclusions': []
        }
        
        # 添加结论
        if abs(ic_rank_corr) < 0.8:
            report['conclusions'].append("IC和RankIC相关性较低，表明存在非线性关系")
        
        if avg_diff > 0.05:
            report['conclusions'].append("IC和RankIC平均差异较大，可能存在异常值影响")
        
        if abs(ic_df['ic_5d'].skew()) > 1.0:
            report['conclusions'].append("IC分布严重偏斜，可能影响Pearson相关系数的稳定性")
        
        if ic_df['ic_5d'].kurtosis() > 3.0:
            report['conclusions'].append("IC分布存在尖峰厚尾特征，对异常值敏感")
        
        # 保存报告
        import json
        with open('factor_results/ic_rankic_detailed_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        print(f"✓ 详细报告已保存: factor_results/ic_rankic_detailed_report.json")
        
        # 打印报告摘要
        print(f"\n{'='*80}")
        print("IC-RankIC差异分析报告摘要")
        print(f"{'='*80}")
        print(f"分析日期: {report['analysis_date']}")
        print(f"总日期数: {report['total_dates']}")
        print(f"时间范围: {report['time_range']}")
        print(f"\nIC统计:")
        print(f"  均值: {ic_mean:.6f}")
        print(f"  标准差: {ic_std:.6f}")
        print(f"  偏度: {report['ic_statistics']['skewness']:.4f}")
        print(f"  峰度: {report['ic_statistics']['kurtosis']:.4f}")
        print(f"\nRankIC统计:")
        print(f"  均值: {rankic_mean:.6f}")
        print(f"  标准差: {rankic_std:.6f}")
        print(f"  偏度: {report['rankic_statistics']['skewness']:.4f}")
        print(f"  峰度: {report['rankic_statistics']['kurtosis']:.4f}")
        print(f"\n差异分析:")
        print(f"  IC-RankIC相关性: {ic_rank_corr:.6f}")
        print(f"  平均绝对差异: {avg_diff:.6f}")
        print(f"  最大绝对差异: {report['difference_analysis']['max_absolute_diff']:.6f}")
        print(f"  差异>0.1的日期数: {report['difference_analysis']['dates_with_large_diff']}")
        
        if report['conclusions']:
            print(f"\n结论:")
            for i, conclusion in enumerate(report['conclusions'], 1):
                print(f"  {i}. {conclusion}")
        else:
            print(f"\n结论: 未发现明显问题")

def main():
    """主函数"""
    print("IC和RankIC差异分析")
    print("="*50)
    
    # 创建分析器
    analyzer = ICvsRankICAnalyzer()
    
    # 加载IC数据
    ic_df = analyzer.load_ic_data()
    
    if ic_df is not None:
        # 分析IC和RankIC差异
        ic_df = analyzer.analyze_ic_rankic_difference(ic_df)
        
        # 生成详细报告
        analyzer.generate_detailed_report(ic_df)
        
        print(f"\n✓ 分析完成！")
    else:
        print("✗ 无法加载IC数据，分析失败")

if __name__ == "__main__":
    main() 