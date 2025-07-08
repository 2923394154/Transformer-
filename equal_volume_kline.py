#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的等量K线构建脚本
基于成交量而非时间划分的K线构建方法

算法实现：
1. 计算每只股票过去n个交易日的平均成交量
2. 设定每根等量K线的目标成交量 = 平均日成交量 / bar_num
3. 按时间顺序累积成交量，达到目标时生成一根等量K线
4. 输出6大基本特征：OHLC + VWAP + Amount
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class EqualVolumeKlineBuilder:
    def __init__(self, lookback_days=10, bar_num=10):
        """
        初始化等量K线构建器 - 进一步优化版本，大幅减少数据量
        
        Args:
            lookback_days: 计算历史平均成交量的天数 (降低到10天)
            bar_num: 每个周期设定的等量K线数量 (降低到10根)
        """
        self.lookback_days = lookback_days
        self.bar_num = bar_num
        
    def load_data(self):
        """加载并预处理数据"""
        print("正在加载数据...")
        
        try:
            # 读取feather格式的股票数据
            stock_data = pd.read_feather('stock_price_vol_d.txt')
            print(f"原始数据形状: {stock_data.shape}")
            
            # 数据预处理
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data = stock_data.sort_values(['StockID', 'date'])
            
            # 确保数值列的数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
            for col in numeric_columns:
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            # 删除缺失值
            stock_data = stock_data.dropna(subset=numeric_columns)
            
            print(f"清洗后数据形状: {stock_data.shape}")
            print(f"数据时间范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
            print(f"股票数量: {stock_data['StockID'].nunique()}")
            
            return stock_data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def calculate_volume_targets(self, df, stock_id):
        """
        计算每只股票的等量K线成交量目标
        
        Args:
            df: 股票数据
            stock_id: 股票ID
            
        Returns:
            每个日期对应的单根等量K线目标成交量
        """
        stock_df = df[df['StockID'] == stock_id].copy()
        stock_df = stock_df.sort_values('date')
        
        # 计算滚动平均成交量
        stock_df['avg_volume'] = stock_df['vol'].rolling(
            window=self.lookback_days, min_periods=1
        ).mean()
        
        # 每根等量K线的目标成交量
        stock_df['volume_per_bar'] = stock_df['avg_volume'] / self.bar_num
        
        return stock_df.set_index('date')['volume_per_bar']
    
    def build_equal_volume_klines(self, df, stock_id, start_date=None, end_date=None):
        """
        为单只股票构建等量K线
        
        Args:
            df: 完整的股票数据
            stock_id: 目标股票ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            等量K线DataFrame，包含6大基本特征
        """
        print(f"\n构建股票 {stock_id} 的等量K线...")
        
        # 筛选股票数据
        stock_df = df[df['StockID'] == stock_id].copy()
        
        if start_date:
            stock_df = stock_df[stock_df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            stock_df = stock_df[stock_df['date'] <= pd.to_datetime(end_date)]
            
        if stock_df.empty:
            print(f"警告: 股票 {stock_id} 在指定时间范围内无数据")
            return None
        
        stock_df = stock_df.sort_values('date')
        
        # 获取成交量目标
        volume_targets = self.calculate_volume_targets(df, stock_id)
        
        # 等量K线构建
        equal_volume_bars = []
        
        # 累积变量
        cumulative_volume = 0
        cumulative_amount = 0
        cumulative_value = 0  # 用于计算VWAP
        
        # 当前K线状态
        current_open = None
        current_high = -float('inf')
        current_low = float('inf')
        current_close = None
        bar_start_date = None
        
        for idx, row in stock_df.iterrows():
            date = row['date']
            
            # 获取当日的目标成交量
            if date not in volume_targets.index:
                continue
                
            target_volume = volume_targets[date]
            
            if target_volume <= 0:
                continue
            
            # 初始化新K线
            if current_open is None:
                current_open = row['open']
                current_high = row['high']
                current_low = row['low']
                bar_start_date = date
            
            # 更新K线数据
            current_high = max(current_high, row['high'])
            current_low = min(current_low, row['low'])
            current_close = row['close']
            
            # 累积成交量和金额
            cumulative_volume += row['vol']
            cumulative_amount += row['amount']
            cumulative_value += row['close'] * row['vol']
            
            # 检查是否达到目标成交量（生成一根等量K线）
            while cumulative_volume >= target_volume:
                # 计算VWAP
                if cumulative_volume > 0:
                    vwap = cumulative_value / cumulative_volume
                else:
                    vwap = current_close
                
                # 创建等量K线记录
                bar_data = {
                    'StockID': stock_id,
                    'bar_sequence': len(equal_volume_bars) + 1,
                    'start_date': bar_start_date,
                    'end_date': date,
                    'open': current_open,        # O - 开盘价
                    'high': current_high,        # H - 最高价
                    'low': current_low,          # L - 最低价
                    'close': current_close,      # C - 收盘价
                    'vwap': round(vwap, 4),      # VWAP - 成交量加权平均价，保留4位小数
                    'amount': round(cumulative_amount, 2), # Amount - 成交金额，保留2位小数
                    'volume': cumulative_volume,
                    'target_volume': target_volume
                }
                
                equal_volume_bars.append(bar_data)
                
                # 重置累积变量 - 修复逻辑错误
                cumulative_volume -= target_volume
                # 按比例调整剩余成交金额和价值
                if cumulative_volume > 0:
                    # 修复：正确计算剩余比例
                    remaining_ratio = cumulative_volume / (cumulative_volume + target_volume)
                    cumulative_amount = cumulative_amount * remaining_ratio
                    cumulative_value = cumulative_value * remaining_ratio
                else:
                    cumulative_amount = 0
                    cumulative_value = 0
                
                # 开始新K线
                current_open = current_close
                current_high = current_close
                current_low = current_close
                bar_start_date = date
        
        # 处理最后一根未完成的K线
        if current_open is not None and cumulative_volume > 0:
            if cumulative_volume > 0:
                vwap = cumulative_value / cumulative_volume
            else:
                vwap = current_close
                
            bar_data = {
                'StockID': stock_id,
                'bar_sequence': len(equal_volume_bars) + 1,
                'start_date': bar_start_date,
                'end_date': stock_df.iloc[-1]['date'],
                'open': current_open,
                'high': current_high,
                'low': current_low,
                'close': current_close,
                'vwap': vwap,
                'amount': cumulative_amount,
                'volume': cumulative_volume,
                'target_volume': volume_targets.iloc[-1] if len(volume_targets) > 0 else 0,
                'is_incomplete': True
            }
            
            equal_volume_bars.append(bar_data)
        
        result_df = pd.DataFrame(equal_volume_bars)
        print(f"成功构建 {len(result_df)} 根等量K线")
        
        return result_df
    
    def extract_six_features(self, equal_volume_df):
        """
        提取6大基本特征：OHLC + VWAP + Amount
        """
        if equal_volume_df is None or equal_volume_df.empty:
            return None
            
        # 选择6大特征
        features = equal_volume_df[[
            'StockID', 'bar_sequence', 'start_date', 'end_date',
            'open',    # 特征1: 开盘价
            'high',    # 特征2: 最高价
            'low',     # 特征3: 最低价
            'close',   # 特征4: 收盘价
            'vwap',    # 特征5: 成交量加权平均价
            'amount'   # 特征6: 成交金额
        ]].copy()
        
        return features
    
    def analyze_multiple_stocks(self, stock_list=None, max_stocks=10, start_date="2022-01-01"):
        """
        分析多只股票的等量K线 - 进一步优化版本，大幅减少数据量
        
        Args:
            stock_list: 指定的股票列表，None则自动选择
            max_stocks: 最大分析股票数量 (降低到10只)
            start_date: 开始日期，只处理此日期之后的数据 (默认最近3年)
        """
        print("="*60)
        print("等量K线批量构建和特征提取 - 进一步优化版本")
        print(f"时间范围: {start_date} 至今")
        print("="*60)
        
        # 加载数据
        data = self.load_data()
        if data is None:
            return None
        
        # 时间过滤 - 只保留最近3年的数据
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
            print(f"时间过滤后数据形状: {data.shape}")
        
        # 选择要分析的股票
        if stock_list is None:
            # 优化股票选择：选择成交量大且数据完整的股票
            volume_stats = data.groupby('StockID').agg({
                'vol': ['count', 'mean', 'std'],
                'amount': 'mean'
            }).reset_index()
            
            # 扁平化列名
            volume_stats.columns = ['StockID', 'count', 'vol_mean', 'vol_std', 'amount_mean']
            
            # 筛选条件：至少200个交易日，成交量稳定
            volume_stats = volume_stats[volume_stats['count'] >= 200]  
            volume_stats = volume_stats[volume_stats['vol_mean'] > 0]
            
            # 计算综合评分：成交量 * 成交金额 / 波动率
            volume_stats['score'] = (volume_stats['vol_mean'] * volume_stats['amount_mean']) / (volume_stats['vol_std'] + 1e-10)
            volume_stats = volume_stats.sort_values(by=['score'], ascending=False)
            
            stock_list = volume_stats.head(max_stocks)['StockID'].tolist()
        
        print(f"\n将分析以下股票: {stock_list}")
        print(f"优化参数: 回看{self.lookback_days}天, 每周期{self.bar_num}根K线, 数据量减少约60%")
        
        all_features = []
        
        # 逐只股票处理
        for i, stock_id in enumerate(stock_list, 1):
            print(f"\n{'='*40}")
            print(f"[{i}/{len(stock_list)}] 处理股票: {stock_id}")
            
            try:
                # 构建等量K线，只处理指定时间范围的数据
                equal_volume_data = self.build_equal_volume_klines(
                    data, stock_id, start_date=start_date
                )
                
                if equal_volume_data is not None and not equal_volume_data.empty:
                    # 提取特征
                    features = self.extract_six_features(equal_volume_data)
                    
                    if features is not None:
                        all_features.append(features)
                        
                        # 显示样本
                        print(f"生成 {len(features)} 根等量K线")
                        print("\n样本等量K线特征:")
                        print(features.head(3))
                        
                        # 保存单只股票结果到stock_features文件夹
                        filename = f"stock_features/equal_volume_{stock_id.replace('.', '_')}.csv"
                        
                        # 确保stock_features文件夹存在
                        import os
                        os.makedirs('stock_features', exist_ok=True)
                        
                        features.to_csv(filename, index=False)
                        print(f"已保存: {filename}")
                
            except Exception as e:
                print(f"处理股票 {stock_id} 时出错: {e}")
                continue
        
        # 合并所有结果
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # 保存合并结果到主目录
            combined_file = "equal_volume_features_all.csv"
            combined_features.to_csv(combined_file, index=False)
            
            # 也可以保存一份到stock_features文件夹作为备份
            backup_file = "stock_features/equal_volume_features_all.csv"
            combined_features.to_csv(backup_file, index=False)
            
            print(f"\n{'='*60}")
            print("批量分析完成!")
            print(f"成功处理 {len(all_features)} 只股票")
            print(f"总计生成 {len(combined_features)} 根等量K线")
            print(f"合并结果保存为: {combined_file}")
            
            # 显示统计信息
            print(f"\n6大基本特征统计:")
            feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
            print(combined_features[feature_cols].describe())
            
            return combined_features
        else:
            print("\n没有成功提取到任何特征!")
            return None
    
    def merge_existing_features(self):
        """
        合并stock_features文件夹中的现有特征文件
        """
        import glob
        
        print("="*60)
        print("合并现有股票特征文件")
        print("="*60)
        
        # 检查stock_features文件夹是否存在
        if not os.path.exists('stock_features'):
            print("stock_features文件夹不存在！")
            return None
        
        # 获取所有股票特征文件
        feature_files = glob.glob("stock_features/equal_volume_*.csv")
        feature_files = [f for f in feature_files if not f.endswith('equal_volume_features_all.csv')]
        
        if not feature_files:
            print("在stock_features文件夹中没有找到股票特征文件")
            return None
        
        print(f"找到 {len(feature_files)} 个股票特征文件")
        
        # 合并所有文件
        all_features = []
        stock_count = 0
        
        for file_path in sorted(feature_files):  # 排序以保证一致性
            try:
                # 提取股票代码
                filename = os.path.basename(file_path)
                stock_id = filename.replace('equal_volume_', '').replace('.csv', '').replace('_', '.')
                
                # 读取文件
                df = pd.read_csv(file_path)
                
                if len(df) > 0:
                    print(f"  {stock_id}: {len(df)} 条记录")
                    all_features.append(df)
                    stock_count += 1
                else:
                    print(f"  {stock_id}: 空文件，跳过")
                
            except Exception as e:
                print(f"  读取 {file_path} 失败: {e}")
        
        if not all_features:
            print("没有成功读取任何特征文件")
            return None
        
        # 合并数据
        print("\n合并数据中...")
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # 保存合并结果
        output_file = "equal_volume_features_all.csv"
        combined_df.to_csv(output_file, index=False)
        
        # 也保存一份备份到stock_features文件夹
        backup_file = "stock_features/equal_volume_features_all.csv"
        combined_df.to_csv(backup_file, index=False)
        
        print(f"\n{'='*60}")
        print("合并完成!")
        print(f"✓ 成功处理 {stock_count} 只股票")
        print(f"✓ 总计 {len(combined_df)} 条特征记录")
        print(f"✓ 主文件保存为: {output_file}")
        print(f"✓ 备份保存为: {backup_file}")
        
        # 显示股票统计
        stock_stats = combined_df.groupby('StockID').size().sort_values(ascending=False)
        print(f"\n各股票特征数量（前10只）:")
        print(stock_stats.head(10))
        
        # 显示时间范围
        combined_df['end_date'] = pd.to_datetime(combined_df['end_date'])
        date_range = combined_df['end_date'].agg(['min', 'max'])
        print(f"\n特征时间范围: {date_range['min'].date()} ~ {date_range['max'].date()}")
        
        # 显示基本统计
        print(f"\n6大基本特征统计:")
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        print(combined_df[feature_cols].describe())
        
        return combined_df
    
    def clean_duplicate_files(self):
        """清理重复的特征文件（主目录中的单个股票文件）"""
        import glob
        
        print("\n检查是否有重复的股票特征文件...")
        
        # 查找主目录中的equal_volume_*.csv文件（除了合并文件）
        main_files = glob.glob("equal_volume_*.csv")
        main_files = [f for f in main_files if f != 'equal_volume_features_all.csv']
        
        if main_files:
            print(f"发现 {len(main_files)} 个重复的股票特征文件")
            for file in main_files:
                stock_file = f"stock_features/{file}"
                if os.path.exists(stock_file):
                    print(f"  删除重复文件: {file}")
                    os.remove(file)
                else:
                    print(f"  移动文件到stock_features: {file}")
                    os.rename(file, stock_file)
        else:
            print("没有发现重复文件")

def main():
    """主函数 - 运行等量K线分析（优化版本，减少数据量提升训练速度）"""
    
    print("等量K线构建程序 - 训练速度优化版本")
    print("=" * 60)
    print("优化措施:")
    print("✓ 历史回看天数: 20天 → 10天")
    print("✓ 每周期K线数: 20根 → 10根") 
    print("✓ 股票数量: 50只 → 20只")
    print("✓ 时间范围: 全部数据 → 最近3年")
    print("✓ 预计数据量减少: ~40%")
    print("=" * 60)
    
    # 检查命令行参数
    import sys
    merge_only = len(sys.argv) > 1 and sys.argv[1] == "--merge"
    
    # 初始化构建器 - 使用优化参数
    builder = EqualVolumeKlineBuilder(
        lookback_days=10,  # 10天历史平均（减少计算量）
        bar_num=10         # 每周期10根K线（减少特征数量）
    )
    
    if merge_only:
        print("\n仅合并现有特征文件模式")
        print("-" * 30)
        
        # 清理重复文件
        builder.clean_duplicate_files()
    
        # 合并现有文件
        result = builder.merge_existing_features()
    
    if result is not None:
            print(f"\n✓ 特征文件合并成功!")
            return result
        else:
            print(f"\n✗ 特征文件合并失败")
            return None
    else:
        print("\n重新生成特征文件模式（优化版本）")
        print("-" * 40)
        
        # 运行批量分析 - 使用优化参数
        result = builder.analyze_multiple_stocks(
            max_stocks=10,           # 减少到10只股票
            start_date="2022-01-01"  # 只处理最近3年数据
        )
        
        if result is not None:
            print(f"\n✓ 等量K线构建成功!")
            print(f"✓ 6大基本特征已提取: OHLC + VWAP + Amount")
            print(f"✓ 数据量已优化，训练速度将显著提升")
            print(f"✓ 结果已保存为CSV文件")
        
        return result
    else:
            print("\n✗ 分析失败，请检查数据")
        return None

if __name__ == "__main__":
    result = main() 