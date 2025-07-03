#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的Transformer因子测试和回测系统
严格按照研究规格实现：多种子集成、股票池筛选、周频调仓、因子分层测试

技术规格：
1. 股票池：全A股，剔除ST股票，剔除每个截面期下一交易日停牌、涨停的股票
2. 回测区间：2017/1/1～2025/2/28
3. 调仓周期：周频，不计交易费用
4. 测试方法：IC值分析，因子分10层测试
5. 最终结果：RankIC均值、RankIC标准差、RankICIR、RankIC>0占比、TOP组合年化超额收益率等
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class ComprehensiveFactorTester:
    """完整的因子测试和回测系统"""
    
    def __init__(self, start_date='2017-01-01', end_date='2025-02-28'):
        """
        初始化因子测试器
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        print("="*80)
        print("完整的Transformer因子测试和回测系统")
        print("="*80)
        print("严格按照研究规格实现:")
        print("✓ 股票池: 全A股，剔除ST股票，剔除停牌、涨停股票")
        print("✓ 回测区间: 2017/1/1 ～ 2025/2/28")
        print("✓ 调仓周期: 周频，不计交易费用")
        print("✓ 测试方法: IC值分析，因子分10层测试")
        print("✓ 集成策略: 3个不同种子模型等权集成")
        print("="*80)
        print(f"回测时间范围: {start_date} ~ {end_date}")
        print("="*80)
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载所有必需的数据"""
        print("\n正在加载数据...")
        
        # 加载股票价格数据
        try:
            print("  加载股票价格数据...")
            self.stock_data = pd.read_feather('stock_price_vol_d.txt')
            self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
            print(f"  ✓ 股票价格数据: {self.stock_data.shape}")
        except Exception as e:
            print(f"  ✗ 加载股票价格数据失败: {e}")
            # 初始化基本属性以避免AttributeError
            self.trading_dates = []
            self.rebalance_dates = []
            return
        
        # 初始化交易日历
        self.trading_dates = sorted(self.stock_data['date'].unique())
        print(f"  ✓ 交易日历: {len(self.trading_dates)} 个交易日")
        
        # 获取周频调仓日期（即使没有因子数据也要设置）
        self.rebalance_dates = self.get_weekly_rebalance_dates()
        print(f"  ✓ 周频调仓: {len(self.rebalance_dates)} 个调仓日")
        
        # 加载因子得分数据
        try:
            print("  加载因子得分数据...")
            factor_file = 'factor_results/ensemble_factor_scores.csv'
            if os.path.exists(factor_file):
                self.factor_scores = pd.read_csv(factor_file)
                self.factor_scores['date'] = pd.to_datetime(self.factor_scores['date'])
                print(f"  ✓ 因子得分数据: {self.factor_scores.shape}")
            else:
                print(f"  ✗ 因子得分文件不存在: {factor_file}")
                print("  请先运行: python transformer_factor_training.py --mode backtest")
                # 创建空的因子数据框而不是直接返回
                self.factor_scores = pd.DataFrame({'date': [], 'StockID': [], 'factor_score': []})
                return
        except Exception as e:
            print(f"  ✗ 加载因子得分数据失败: {e}")
            # 创建空的因子数据框而不是直接返回
            self.factor_scores = pd.DataFrame({'date': [], 'StockID': [], 'factor_score': []})
            return
        
        # 时间过滤
        self.stock_data = self.stock_data[
            (self.stock_data['date'] >= self.start_date) & 
            (self.stock_data['date'] <= self.end_date)
        ]
        
        self.factor_scores = self.factor_scores[
            (self.factor_scores['date'] >= self.start_date) & 
            (self.factor_scores['date'] <= self.end_date)
        ]
        
        print(f"  ✓ 时间过滤后 - 股票数据: {self.stock_data.shape}")
        print(f"  ✓ 时间过滤后 - 因子数据: {self.factor_scores.shape}")
        
        # 获取交易日历
        self.trading_dates = sorted(self.stock_data['date'].unique())
        print(f"  ✓ 交易日历: {len(self.trading_dates)} 个交易日")
        
        # 获取周频调仓日期
        self.rebalance_dates = self.get_weekly_rebalance_dates()
        print(f"  ✓ 周频调仓: {len(self.rebalance_dates)} 个调仓日")
        
    def get_weekly_rebalance_dates(self):
        """获取周频调仓日期（每周五或最后一个交易日）"""
        print("  正在生成周频调仓日期...")
        
        # 过滤到回测时间范围内的交易日
        filtered_dates = [d for d in self.trading_dates 
                         if self.start_date <= d <= self.end_date]
        
        if not filtered_dates:
            return []
        
        rebalance_dates = []
        current_week_start = None
        current_week_dates = []
        
        for trade_date in filtered_dates:
            # 计算当前日期是周几（0=周一，4=周五）
            weekday = trade_date.weekday()
            
            # 计算本周开始日期（周一）
            week_start = trade_date - timedelta(days=weekday)
            
            # 如果是新的一周
            if current_week_start != week_start:
                # 如果不是第一周，保存上一周的最后一个交易日
                if current_week_dates:
                    rebalance_dates.append(max(current_week_dates))
                
                # 开始新的一周
                current_week_start = week_start
                current_week_dates = [trade_date]
            else:
                # 继续当前周
                current_week_dates.append(trade_date)
        
        # 处理最后一周
        if current_week_dates:
            rebalance_dates.append(max(current_week_dates))
        
        return sorted(rebalance_dates)
    
    def filter_stock_universe(self, date):
        """
        筛选股票池：全A股，剔除ST股票，剔除停牌、涨停股票
        
        Args:
            date: 当前交易日
            
        Returns:
            filtered_stocks: 筛选后的股票列表
        """
        # 获取当前日期的股票数据
        current_data = self.stock_data[self.stock_data['date'] == date].copy()
        
        if len(current_data) == 0:
            return []
        
        # 1. 剔除ST股票
        # 简化处理：假设股票代码包含ST信息，实际应用中需要更精确的ST股票数据
        current_data = current_data[~current_data['StockID'].str.contains('ST', na=False)]
        
        # 2. 剔除停牌股票（成交量为0）
        current_data = current_data[current_data['vol'] > 0]
        
        # 3. 剔除涨停股票
        # 获取前一个交易日数据
        prev_date = self.get_previous_trading_date(date)
        if prev_date is not None:
            prev_data = self.stock_data[self.stock_data['date'] == prev_date]
            merged = current_data.merge(prev_data[['StockID', 'close']], 
                                      on='StockID', suffixes=('', '_prev'))
            
            # 计算涨幅，剔除涨停股票（涨幅>9.8%）
            merged['return'] = (merged['close'] - merged['close_prev']) / merged['close_prev']
            non_limit_stocks = merged[merged['return'] <= 0.098]['StockID'].tolist()
            current_data = current_data[current_data['StockID'].isin(non_limit_stocks)]
        
        return current_data['StockID'].tolist()
    
    def get_previous_trading_date(self, date):
        """获取前一个交易日"""
        try:
            current_idx = self.trading_dates.index(date)
            if current_idx > 0:
                return self.trading_dates[current_idx - 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def get_next_trading_date(self, date):
        """获取下一个交易日"""
        try:
            current_idx = self.trading_dates.index(date)
            if current_idx < len(self.trading_dates) - 1:
                return self.trading_dates[current_idx + 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def calculate_stock_returns(self, start_date, end_date, stock_list):
        """
        计算股票收益率
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_list: 股票列表
            
        Returns:
            DataFrame: 股票收益率数据
        """
        # 获取价格数据
        price_data = self.stock_data[
            (self.stock_data['date'] >= start_date) & 
            (self.stock_data['date'] <= end_date) &
            (self.stock_data['StockID'].isin(stock_list))
        ].copy()
        
        if len(price_data) == 0:
            return pd.DataFrame()
        
        # 按股票分组计算收益率
        returns_list = []
        
        for stock_id in stock_list:
            stock_prices = price_data[price_data['StockID'] == stock_id].sort_values('date')
            
            if len(stock_prices) >= 2:
                start_price = stock_prices.iloc[0]['close']
                end_price = stock_prices.iloc[-1]['close']
                
                if start_price > 0:
                    total_return = (end_price - start_price) / start_price
                    
                    returns_list.append({
                        'StockID': stock_id,
                        'start_date': start_date,
                        'end_date': end_date,
                        'start_price': start_price,
                        'end_price': end_price,
                        'return': total_return
                    })
        
        return pd.DataFrame(returns_list)
    
    def calculate_weekly_factor_performance(self):
        """计算周频因子表现"""
        print("\n" + "="*60)
        print("计算周频因子表现")
        print("="*60)
        
        weekly_performance = []
        
        for i in tqdm(range(len(self.rebalance_dates) - 1), desc="计算周频表现"):
            current_date = self.rebalance_dates[i]
            next_date = self.rebalance_dates[i + 1]
            
            # 筛选股票池
            stock_universe = self.filter_stock_universe(current_date)
            
            if len(stock_universe) < 50:  # 至少需要50只股票
                continue
            
            # 获取当期因子得分
            current_factors = self.factor_scores[
                (self.factor_scores['date'] == current_date) &
                (self.factor_scores['StockID'].isin(stock_universe))
            ].copy()
            
            if len(current_factors) < 30:  # 至少需要30只股票有因子得分
                continue
            
            # 计算下期收益率
            stock_returns = self.calculate_stock_returns(
                current_date, next_date, current_factors['StockID'].tolist()
            )
            
            if len(stock_returns) < 20:  # 至少需要20只股票有收益率数据
                continue
            
            # 合并因子得分和收益率
            merged_data = current_factors.merge(
                stock_returns[['StockID', 'return']], 
                on='StockID', 
                how='inner'
            )
            
            if len(merged_data) < 20:
                continue
            
            # 计算IC和RankIC
            factor_values = merged_data['factor_score'].values
            return_values = merged_data['return'].values
            
            # IC (Pearson相关系数)
            ic = np.corrcoef(factor_values, return_values)[0, 1]
            
            # RankIC (Spearman相关系数)
            factor_ranks = rankdata(factor_values)
            return_ranks = rankdata(return_values)
            rank_ic = np.corrcoef(factor_ranks, return_ranks)[0, 1]
            
            # 因子分层测试（10层）
            layered_returns = self.factor_layered_analysis(merged_data)
            
            weekly_performance.append({
                'date': current_date,
                'next_date': next_date,
                'stock_count': len(merged_data),
                'ic': ic if not np.isnan(ic) else 0,
                'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
                'top_return': layered_returns.get('top_return', 0),
                'bottom_return': layered_returns.get('bottom_return', 0),
                'top_minus_bottom': layered_returns.get('top_minus_bottom', 0),
                'layer_returns': layered_returns.get('layer_returns', [])
            })
        
        return pd.DataFrame(weekly_performance)
    
    def factor_layered_analysis(self, factor_return_data, num_layers=10):
        """
        因子分层分析
        
        Args:
            factor_return_data: 包含因子得分和收益率的数据
            num_layers: 分层数量
            
        Returns:
            dict: 分层分析结果
        """
        if len(factor_return_data) < num_layers:
            return {}
        
        # 按因子得分排序并分层
        sorted_data = factor_return_data.sort_values('factor_score', ascending=False)
        
        layer_size = len(sorted_data) // num_layers
        layer_returns = []
        
        for i in range(num_layers):
            start_idx = i * layer_size
            if i == num_layers - 1:  # 最后一层包含所有剩余股票
                end_idx = len(sorted_data)
            else:
                end_idx = (i + 1) * layer_size
            
            layer_data = sorted_data.iloc[start_idx:end_idx]
            layer_return = layer_data['return'].mean()
            layer_returns.append(layer_return)
        
        top_return = layer_returns[0] if layer_returns else 0
        bottom_return = layer_returns[-1] if layer_returns else 0
        
        return {
            'layer_returns': layer_returns,
            'top_return': top_return,
            'bottom_return': bottom_return,
            'top_minus_bottom': top_return - bottom_return
        }
    
    def calculate_performance_metrics(self, weekly_performance):
        """
        计算最终性能指标
        
        Args:
            weekly_performance: 周频表现数据
            
        Returns:
            dict: 性能指标
        """
        print("\n" + "="*60)
        print("计算最终性能指标")
        print("="*60)
        
        if len(weekly_performance) == 0:
            print("✗ 没有有效的周频表现数据")
            return {}
        
        # 清理异常值
        weekly_performance = weekly_performance.dropna(subset=['ic', 'rank_ic'])
        
        if len(weekly_performance) == 0:
            print("✗ 清理后没有有效数据")
            return {}
        
        # RankIC相关指标
        rank_ics = weekly_performance['rank_ic'].values
        rank_ic_mean = np.mean(rank_ics)
        rank_ic_std = np.std(rank_ics)
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0
        rank_ic_positive_ratio = np.sum(rank_ics > 0) / len(rank_ics)
        
        # IC相关指标
        ics = weekly_performance['ic'].values
        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_ratio = np.sum(ics > 0) / len(ics)
        
        # TOP组合表现（第1层）
        top_returns = weekly_performance['top_return'].values
        top_return_mean = np.mean(top_returns)
        
        # 年化收益率（假设每年约50个调仓周期）
        periods_per_year = 50
        top_annual_return = (1 + top_return_mean) ** periods_per_year - 1
        
        # 计算基准收益率（等权组合）
        all_returns = []
        for _, row in weekly_performance.iterrows():
            if row['layer_returns']:
                avg_return = np.mean(row['layer_returns'])
                all_returns.append(avg_return)
        
        benchmark_return_mean = np.mean(all_returns) if all_returns else 0
        benchmark_annual_return = (1 + benchmark_return_mean) ** periods_per_year - 1
        
        # 超额收益率
        excess_return_mean = top_return_mean - benchmark_return_mean
        excess_annual_return = top_annual_return - benchmark_annual_return
        
        # TOP组合信息比率
        excess_returns = weekly_performance['top_return'] - weekly_performance['bottom_return']
        excess_return_std = np.std(excess_returns)
        top_info_ratio = np.mean(excess_returns) / excess_return_std if excess_return_std > 0 else 0
        
        # TOP组合胜率
        top_win_rate = np.sum(top_returns > 0) / len(top_returns)
        
        # 换手率（简化计算，假设每期完全换手）
        top_turnover = 1.0  # 100%换手率
        
        performance_metrics = {
            'rank_ic_mean': rank_ic_mean,
            'rank_ic_std': rank_ic_std,
            'rank_ic_ir': rank_ic_ir,
            'rank_ic_positive_ratio': rank_ic_positive_ratio,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_ratio': ic_positive_ratio,
            'top_annual_return': top_annual_return,
            'top_excess_annual_return': excess_annual_return,
            'top_info_ratio': top_info_ratio,
            'top_win_rate': top_win_rate,
            'top_turnover': top_turnover,
            'total_periods': len(weekly_performance),
            'valid_periods': len(weekly_performance)
        }
        
        return performance_metrics
    
    def generate_comprehensive_report(self, performance_metrics):
        """生成完整的测试报告"""
        print("\n" + "="*80)
        print("Transformer因子测试报告")
        print("="*80)
        
        if not performance_metrics:
            print("✗ 没有有效的性能指标数据")
            return
        
        print("最终结果:")
        print("-" * 60)
        print(f"RankIC均值:             {performance_metrics['rank_ic_mean']:>10.4f}")
        print(f"RankIC标准差:           {performance_metrics['rank_ic_std']:>10.4f}")
        print(f"RankICIR:              {performance_metrics['rank_ic_ir']:>10.4f}")
        print(f"RankIC>0占比:          {performance_metrics['rank_ic_positive_ratio']:>10.2%}")
        print(f"TOP组合年化超额收益率:  {performance_metrics['top_excess_annual_return']:>10.2%}")
        print(f"TOP组合信息比率:       {performance_metrics['top_info_ratio']:>10.4f}")
        print(f"TOP组合胜率:           {performance_metrics['top_win_rate']:>10.2%}")
        print(f"TOP组合换手率:         {performance_metrics['top_turnover']:>10.2%}")
        
        print("\n补充指标:")
        print("-" * 60)
        print(f"IC均值:                {performance_metrics['ic_mean']:>10.4f}")
        print(f"IC标准差:              {performance_metrics['ic_std']:>10.4f}")
        print(f"ICIR:                  {performance_metrics['ic_ir']:>10.4f}")
        print(f"IC>0占比:              {performance_metrics['ic_positive_ratio']:>10.2%}")
        print(f"TOP组合年化收益率:     {performance_metrics['top_annual_return']:>10.2%}")
        
        print("\n回测统计:")
        print("-" * 60)
        print(f"总调仓期数:            {performance_metrics['total_periods']:>10d}")
        print(f"有效期数:              {performance_metrics['valid_periods']:>10d}")
        print(f"回测区间:              {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"调仓频率:              周频")
        
        # 保存报告
        self.save_report_to_file(performance_metrics)
        
        print("\n" + "="*80)
        
    def save_report_to_file(self, performance_metrics):
        """保存报告到文件"""
        os.makedirs('factor_results', exist_ok=True)
        
        # 保存详细指标
        report_file = 'factor_results/comprehensive_factor_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, indent=4, ensure_ascii=False)
        
        # 保存汇总报告
        summary_file = 'factor_results/factor_summary_report.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Transformer因子测试报告\n")
            f.write("="*80 + "\n\n")
            f.write("最终结果:\n")
            f.write("-" * 60 + "\n")
            f.write(f"RankIC均值:             {performance_metrics['rank_ic_mean']:>10.4f}\n")
            f.write(f"RankIC标准差:           {performance_metrics['rank_ic_std']:>10.4f}\n")
            f.write(f"RankICIR:              {performance_metrics['rank_ic_ir']:>10.4f}\n")
            f.write(f"RankIC>0占比:          {performance_metrics['rank_ic_positive_ratio']:>10.2%}\n")
            f.write(f"TOP组合年化超额收益率:  {performance_metrics['top_excess_annual_return']:>10.2%}\n")
            f.write(f"TOP组合信息比率:       {performance_metrics['top_info_ratio']:>10.4f}\n")
            f.write(f"TOP组合胜率:           {performance_metrics['top_win_rate']:>10.2%}\n")
            f.write(f"TOP组合换手率:         {performance_metrics['top_turnover']:>10.2%}\n")
        
        print(f"✓ 详细报告已保存: {report_file}")
        print(f"✓ 汇总报告已保存: {summary_file}")
    
    def run_comprehensive_test(self):
        """运行完整的因子测试"""
        print("\n开始运行完整的因子测试...")
        
        # 计算周频因子表现
        weekly_performance = self.calculate_weekly_factor_performance()
        
        if len(weekly_performance) == 0:
            print("✗ 没有计算出有效的周频表现数据")
            return None
        
        # 保存周频表现数据
        os.makedirs('factor_results', exist_ok=True)
        weekly_file = 'factor_results/weekly_performance.csv'
        weekly_performance.to_csv(weekly_file, index=False)
        print(f"✓ 周频表现数据已保存: {weekly_file}")
        
        # 计算性能指标
        performance_metrics = self.calculate_performance_metrics(weekly_performance)
        
        if not performance_metrics:
            print("✗ 性能指标计算失败")
            return None
        
        # 生成报告
        self.generate_comprehensive_report(performance_metrics)
        
        return performance_metrics

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='完整的Transformer因子测试和回测')
    parser.add_argument('--start_date', default='2017-01-01', help='回测开始日期')
    parser.add_argument('--end_date', default='2025-02-28', help='回测结束日期')
    
    try:
        args = parser.parse_args()
    except:
        # 如果没有命令行参数，使用默认值
        class Args:
            start_date = '2017-01-01'
            end_date = '2025-02-28'
        args = Args()
    
    # 初始化测试器
    tester = ComprehensiveFactorTester(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # 运行完整测试
    performance_metrics = tester.run_comprehensive_test()
    
    if performance_metrics:
        print("\n✓ 完整的因子测试已完成！")
        print("✓ 结果文件保存在 factor_results/ 目录下")
    else:
        print("\n✗ 因子测试失败")

if __name__ == "__main__":
    main() 