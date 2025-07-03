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
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class EqualVolumeKlineBuilder:
    def __init__(self, lookback_days=20, bar_num=20):
        """
        初始化等量K线构建器
        
        Args:
            lookback_days: 计算历史平均成交量的天数
            bar_num: 每个周期设定的等量K线数量
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
                    'vwap': vwap,               # VWAP - 成交量加权平均价
                    'amount': cumulative_amount, # Amount - 成交金额
                    'volume': cumulative_volume,
                    'target_volume': target_volume
                }
                
                equal_volume_bars.append(bar_data)
                
                # 重置累积变量
                cumulative_volume -= target_volume
                # 按比例调整剩余成交金额和价值
                if cumulative_volume > 0:
                    ratio = cumulative_volume / (cumulative_volume + target_volume)
                    cumulative_amount *= ratio
                    cumulative_value *= ratio
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
    
    def analyze_multiple_stocks(self, stock_list=None, max_stocks=5):
        """
        分析多只股票的等量K线
        
        Args:
            stock_list: 指定的股票列表，None则自动选择
            max_stocks: 最大分析股票数量
        """
        print("="*60)
        print("等量K线批量构建和特征提取")
        print("="*60)
        
        # 加载数据
        data = self.load_data()
        if data is None:
            return None
        
        # 选择要分析的股票
        if stock_list is None:
            # 选择成交量较大的股票
            volume_stats = data.groupby('StockID')['vol'].agg(['count', 'mean']).reset_index()
            volume_stats = volume_stats[volume_stats['count'] >= 100]  # 至少100个交易日
            volume_stats = volume_stats.sort_values(['mean'], ascending=False)
            stock_list = volume_stats.head(max_stocks)['StockID'].tolist()
        
        print(f"\n将分析以下股票: {stock_list}")
        print(f"分析参数: 回看{self.lookback_days}天, 每周期{self.bar_num}根K线")
        
        all_features = []
        
        # 逐只股票处理
        for i, stock_id in enumerate(stock_list, 1):
            print(f"\n{'='*40}")
            print(f"[{i}/{len(stock_list)}] 处理股票: {stock_id}")
            
            try:
                # 构建等量K线
                equal_volume_data = self.build_equal_volume_klines(data, stock_id)
                
                if equal_volume_data is not None and not equal_volume_data.empty:
                    # 提取特征
                    features = self.extract_six_features(equal_volume_data)
                    
                    if features is not None:
                        all_features.append(features)
                        
                        # 显示样本
                        print("\n样本等量K线特征:")
                        print(features.head(3))
                        
                        # 保存单只股票结果
                        filename = f"equal_volume_{stock_id.replace('.', '_')}.csv"
                        features.to_csv(filename, index=False)
                        print(f"已保存: {filename}")
                
            except Exception as e:
                print(f"处理股票 {stock_id} 时出错: {e}")
                continue
        
        # 合并所有结果
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # 保存合并结果
            combined_file = "equal_volume_features_all.csv"
            combined_features.to_csv(combined_file, index=False)
            
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

def main():
    """主函数 - 运行等量K线分析"""
    
    print("等量K线构建程序")
    print("=" * 50)
    
    # 初始化构建器
    builder = EqualVolumeKlineBuilder(
        lookback_days=20,  # 20天历史平均
        bar_num=20         # 每周期20根K线
    )
    
    # 运行批量分析
    result = builder.analyze_multiple_stocks(max_stocks=3)
    
    if result is not None:
        print(f"\n等量K线构建成功!")
        print(f"6大基本特征已提取: OHLC + VWAP + Amount")
        print(f"结果已保存为CSV文件")
        
        return result
    else:
        print("\n分析失败，请检查数据")
        return None

if __name__ == "__main__":
    result = main() 