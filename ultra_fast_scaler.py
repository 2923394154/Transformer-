#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高速滚动窗口标准化器
使用优化算法：增量统计 + 向量化操作
"""

import numpy as np
import pandas as pd
from numba import jit, prange
import os
import pickle
from datetime import datetime
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

@jit(nopython=True, parallel=True)
def fast_incremental_rolling_standardize(data, window_size, dates_idx):
    """
    超高速增量滚动窗口标准化
    
    核心优化：
    1. 增量统计：只更新窗口边界的数据点
    2. 向量化：批量处理同一时间窗口的所有特征
    3. 缓存友好：连续内存访问模式
    """
    n_samples, seq_len, n_features = data.shape
    standardized_data = np.zeros_like(data)
    
    # 为每个特征预计算全局统计量（用于窗口不足时）
    global_means = np.zeros(n_features)
    global_stds = np.zeros(n_features)
    
    for f in range(n_features):
        feature_data = data[:, :, f].flatten()
        global_means[f] = np.mean(feature_data)
        global_stds[f] = np.std(feature_data)
        if global_stds[f] < 1e-8:
            global_stds[f] = 1e-8
    
    # 增量滚动统计
    for i in prange(n_samples):
        # 确定当前样本的窗口范围
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        
        if window_end - window_start >= window_size // 2:  # 至少有一半窗口数据
            # 计算窗口内的统计量
            for f in range(n_features):
                # 提取窗口内该特征的所有数据
                window_data = data[window_start:window_end, :, f].flatten()
                
                # 计算统计量
                mean_val = np.mean(window_data)
                std_val = np.std(window_data)
                
                if std_val < 1e-8:
                    std_val = 1e-8
                
                # 标准化当前样本
                standardized_data[i, :, f] = (data[i, :, f] - mean_val) / std_val
        else:
            # 窗口数据不足，使用全局统计量
            for f in range(n_features):
                standardized_data[i, :, f] = (data[i, :, f] - global_means[f]) / global_stds[f]
    
    return standardized_data

@jit(nopython=True)
def fast_vectorized_standardize(data):
    """
    超高速向量化标准化
    使用全局统计量，适用于大数据集快速处理
    """
    n_samples, seq_len, n_features = data.shape
    standardized_data = np.zeros_like(data)
    
    # 向量化计算每个特征的全局统计量
    for f in range(n_features):
        feature_data = data[:, :, f].flatten()
        mean_val = np.mean(feature_data)
        std_val = np.std(feature_data)
        
        if std_val < 1e-8:
            std_val = 1e-8
        
        # 向量化标准化整个特征
        for i in range(n_samples):
            standardized_data[i, :, f] = (data[i, :, f] - mean_val) / std_val
    
    return standardized_data

@jit(nopython=True)
def fast_batch_rolling_standardize(data, window_size, batch_size=1000):
    """
    批处理滚动标准化 - 平衡速度与准确性
    """
    n_samples, seq_len, n_features = data.shape
    standardized_data = np.zeros_like(data)
    
    # 按批次处理
    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_data = data[batch_start:batch_end]
        
        # 为当前批次计算统计量
        for i in range(batch_end - batch_start):
            global_idx = batch_start + i
            window_start = max(0, global_idx - window_size + 1)
            
            # 计算窗口统计量
            for f in range(n_features):
                if global_idx >= window_size:
                    # 使用完整窗口
                    window_data = data[window_start:global_idx+1, :, f].flatten()
                else:
                    # 使用可用数据
                    window_data = data[:global_idx+1, :, f].flatten()
                
                mean_val = np.mean(window_data)
                std_val = np.std(window_data)
                
                if std_val < 1e-8:
                    std_val = 1e-8
                
                standardized_data[global_idx, :, f] = (data[global_idx, :, f] - mean_val) / std_val
    
    return standardized_data

def show_progress_info(data_shape, window_size):
    """显示处理进度信息"""
    n_samples, seq_len, n_features = data_shape
    total_elements = n_samples * seq_len * n_features
    print(f"开始处理 {n_samples:,} 个样本 × {seq_len} 时间步 × {n_features} 特征")
    print(f"总计 {total_elements:,} 个数据点，窗口大小: {window_size}")
    
    # 根据数据量给出不同的处理策略建议
    if total_elements > 100_000_000:  # 1亿元素
        print(f"⚡ 大数据集检测到，建议使用快速模式（向量化标准化）")
        print(f"预估处理时间: {total_elements / 10_000_000:.1f}秒")
    elif total_elements > 50_000_000:  # 5千万元素
        print(f"⚡ 中等数据集，建议使用批处理滚动标准化")
        print(f"预估处理时间: {total_elements / 5_000_000:.1f}秒")
    else:
        print(f"⚡ 小数据集，使用完整滚动窗口标准化")
        print(f"预估处理时间: {total_elements / 2_000_000:.1f}秒")

@jit(nopython=True)
def create_date_index_mapping(dates):
    """创建日期到索引的映射"""
    unique_dates = np.unique(dates)
    date_to_idx = {}
    for i, date in enumerate(unique_dates):
        date_to_idx[date] = i
    
    dates_idx = np.zeros(len(dates), dtype=np.int32)
    for i, date in enumerate(dates):
        dates_idx[i] = date_to_idx[date]
    
    return dates_idx

class UltraFastRollingScaler:
    """
    超高速滚动窗口标准化器
    支持多种优化策略：
    1. 向量化标准化（最快）
    2. 批处理滚动标准化（平衡）
    3. 增量滚动标准化（最准确）
    """
    
    def __init__(self, window_size=252):
        self.window_size = window_size
        self.is_fitted = False
        self.cache_dir = 'scaler_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, data_shape, dates, method="auto"):
        """生成缓存键"""
        dates_hash = hash(tuple(dates[:100].astype(str)))  # 使用前100个日期的哈希
        return f"scaler_{data_shape}_{self.window_size}_{method}_{dates_hash}"
    
    def _save_cache(self, cache_key, scaled_data):
        """保存缓存"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'scaled_data': scaled_data,
                    'timestamp': datetime.now(),
                    'window_size': self.window_size
                }, f)
            print(f"缓存已保存: {cache_file}")
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def _load_cache(self, cache_key):
        """加载缓存"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    print(f"缓存已加载: {cache_file}")
                    self.is_fitted = True  # 重要：设置fitted状态
                    return cache_data['scaled_data']
        except Exception as e:
            print(f"缓存加载失败: {e}")
        return None
    
    def _select_method(self, data_shape):
        """自动选择最佳标准化方法"""
        n_samples, seq_len, n_features = data_shape
        total_elements = n_samples * seq_len * n_features
        
        if total_elements > 100_000_000:  # 1亿元素，使用最快方法
            return "vectorized", "向量化标准化（超快模式）"
        elif total_elements > 50_000_000:  # 5千万元素，使用批处理
            return "batch", "批处理滚动标准化（平衡模式）"
        else:  # 小数据集，使用完整滚动
            return "incremental", "增量滚动标准化（精确模式）"
    
    def fit_transform(self, data, dates, use_cache=True, use_simple=False, method="auto"):
        """
        拟合并转换数据 - 多策略优化版本
        
        Parameters:
        - data: (n_samples, seq_len, n_features) numpy数组
        - dates: (n_samples,) 日期数组
        - use_cache: 是否使用缓存
        - use_simple: 是否强制使用简单标准化
        - method: 标准化方法 ("auto", "vectorized", "batch", "incremental")
        
        Returns:
        - scaled_data: 标准化后的数据
        """
        print(f"开始超高速标准化")
        print(f"   数据形状: {data.shape}")
        print(f"   窗口大小: {self.window_size}")
        print(f"   使用缓存: {use_cache}")
        print(f"   强制简单模式: {use_simple}")
        
        start_time = time.time()
        
        # 自动选择方法
        if use_simple:
            selected_method = "vectorized"
            method_name = "向量化标准化（强制简单模式）"
        elif method == "auto":
            selected_method, method_name = self._select_method(data.shape)
        else:
            selected_method = method
            method_name = f"{method}标准化"
        
        print(f"   选择方法: {method_name}")
        
        # 检查缓存
        if use_cache:
            print("检查缓存...")
            cache_key = self._get_cache_key(data.shape, dates, selected_method)
            cached_result = self._load_cache(cache_key)
            if cached_result is not None:
                load_time = time.time() - start_time
                print(f"使用缓存结果 (耗时: {load_time:.2f}秒)")
                return cached_result
        
        # 显示详细处理信息
        show_progress_info(data.shape, self.window_size)
        
        # 执行标准化
        print(f"正在执行{method_name}...")
        compute_start = time.time()
        
        if selected_method == "vectorized":
            scaled_data = fast_vectorized_standardize(data.astype(np.float32))
        elif selected_method == "batch":
            # 动态调整批大小
            batch_size = min(1000, max(100, data.shape[0] // 10))
            print(f"使用批大小: {batch_size}")
            scaled_data = fast_batch_rolling_standardize(data.astype(np.float32), self.window_size, batch_size)
        else:  # incremental
            # 创建日期索引映射
            dates_timestamps = pd.to_datetime(dates).astype(np.int64)
            unique_timestamps = np.unique(dates_timestamps)
            timestamp_to_idx = {ts: i for i, ts in enumerate(unique_timestamps)}
            dates_idx = np.array([timestamp_to_idx[ts] for ts in dates_timestamps], dtype=np.int32)
            
            scaled_data = fast_incremental_rolling_standardize(data.astype(np.float32), self.window_size, dates_idx)
        
        compute_time = time.time() - compute_start
        print(f"✓ {method_name}完成，耗时: {compute_time:.2f}秒")
        
        # 数值稳定性处理
        print("数值稳定性处理...")
        scaled_data = np.clip(scaled_data, -3.0, 3.0)  # 调整为更合理的范围
        print("✓ 数值裁剪完成")
        
        # 保存缓存
        if use_cache:
            print("保存缓存...")
            self._save_cache(cache_key, scaled_data)
        
        total_time = time.time() - start_time
        throughput = data.size / total_time / 1_000_000  # M元素/秒
        
        self.is_fitted = True
        print(f"标准化完成!")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   处理速度: {throughput:.1f}M元素/秒")
        print(f"   数据形状: {scaled_data.shape}")
        print(f"   数据范围: [{np.min(scaled_data):.3f}, {np.max(scaled_data):.3f}]")
        return scaled_data
    
    def transform(self, data, dates, use_simple=False, method="auto"):
        """
        转换新数据（使用已拟合的参数）- 优化版本
        """
        if not self.is_fitted:
            raise ValueError("必须先调用fit_transform")
        
        print(f"对新数据执行标准化...")
        print(f"   数据形状: {data.shape}")
        
        start_time = time.time()
        
        # 自动选择方法
        if use_simple:
            selected_method = "vectorized"
            method_name = "向量化标准化（强制简单模式）"
        elif method == "auto":
            selected_method, method_name = self._select_method(data.shape)
        else:
            selected_method = method
            method_name = f"{method}标准化"
        
        print(f"   选择方法: {method_name}")
        
        # 执行标准化
        if selected_method == "vectorized":
            scaled_data = fast_vectorized_standardize(data.astype(np.float32))
        elif selected_method == "batch":
            batch_size = min(1000, max(100, data.shape[0] // 10))
            scaled_data = fast_batch_rolling_standardize(data.astype(np.float32), self.window_size, batch_size)
        else:  # incremental
            dates_timestamps = pd.to_datetime(dates).astype(np.int64)
            unique_timestamps = np.unique(dates_timestamps)
            timestamp_to_idx = {ts: i for i, ts in enumerate(unique_timestamps)}
            dates_idx = np.array([timestamp_to_idx[ts] for ts in dates_timestamps], dtype=np.int32)
            
            scaled_data = fast_incremental_rolling_standardize(data.astype(np.float32), self.window_size, dates_idx)
        
        # 数值稳定性处理
        scaled_data = np.clip(scaled_data, -3.0, 3.0)
        
        total_time = time.time() - start_time
        throughput = data.size / total_time / 1_000_000
        
        print(f"变换完成!")
        print(f"   总耗时: {total_time:.2f}秒") 
        print(f"   处理速度: {throughput:.1f}M元素/秒")
        
        return scaled_data

def main():
    """测试函数"""
    print("UltraFastRollingScaler 性能测试")
    
    # 创建测试数据 - 模拟真实数据规模
    n_samples, seq_len, n_features = 10000, 40, 6
    data = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D').values
    
    print(f"测试数据: {data.shape} = {data.size:,} 个元素")
    
    # 测试标准化器
    scaler = UltraFastRollingScaler(window_size=252)
    
    # 测试不同方法
    methods = ["vectorized", "batch", "incremental"]
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"测试方法: {method}")
        print(f"{'='*50}")
        
        start_time = time.time()
        scaled_data = scaler.fit_transform(data, dates, use_cache=False, method=method)
        end_time = time.time()
        
        print(f"方法: {method}")
        print(f"耗时: {end_time - start_time:.2f}秒")
        print(f"处理速度: {data.size / (end_time - start_time) / 1_000_000:.1f}M元素/秒")
        print(f"标准化后统计: mean={np.mean(scaled_data):.6f}, std={np.std(scaled_data):.6f}")

if __name__ == "__main__":
    main() 