#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新训练结果回测分析脚本
使用新生成的direct_features数据和训练好的模型进行回测
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
import math
import numba
from numba import jit, prange
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Numba加速的IC计算函数
@jit(nopython=True, parallel=True)
def fast_ic_calculation(factor_scores, returns):
    """使用Numba加速的IC计算"""
    n = len(factor_scores)
    if n < 2:
        return 0.0
    
    factor_mean = 0.0
    return_mean = 0.0
    for i in prange(n):
        factor_mean += factor_scores[i]
        return_mean += returns[i]
    factor_mean /= n
    return_mean /= n
    
    cov = 0.0
    factor_var = 0.0
    return_var = 0.0
    
    for i in prange(n):
        factor_diff = factor_scores[i] - factor_mean
        return_diff = returns[i] - return_mean
        cov += factor_diff * return_diff
        factor_var += factor_diff * factor_diff
        return_var += return_diff * return_diff
    
    denominator = np.sqrt(factor_var * return_var)
    if denominator == 0:
        return 0.0
    return cov / denominator

@jit(nopython=True, parallel=True)
def fast_rankic_calculation(factor_scores, returns):
    """使用Numba加速的RankIC计算"""
    n = len(factor_scores)
    if n < 2:
        return 0.0
    
    factor_ranks = np.zeros(n, dtype=np.int32)
    return_ranks = np.zeros(n, dtype=np.int32)
    
    for i in prange(n):
        factor_rank = 0
        return_rank = 0
        for j in range(n):
            if factor_scores[j] < factor_scores[i]:
                factor_rank += 1
            if returns[j] < returns[i]:
                return_rank += 1
        factor_ranks[i] = factor_rank
        return_ranks[i] = return_rank
    
    factor_mean = 0.0
    return_mean = 0.0
    for i in prange(n):
        factor_mean += factor_ranks[i]
        return_mean += return_ranks[i]
    factor_mean /= n
    return_mean /= n
    
    cov = 0.0
    factor_var = 0.0
    return_var = 0.0
    
    for i in prange(n):
        factor_diff = factor_ranks[i] - factor_mean
        return_diff = return_ranks[i] - return_mean
        cov += factor_diff * return_diff
        factor_var += factor_diff * factor_diff
        return_var += return_diff * return_diff
    
    denominator = np.sqrt(factor_var * return_var)
    if denominator == 0:
        return 0.0
    return cov / denominator

class TransformerSeq2SeqModel(nn.Module):
    """Encoder-Decoder结构，支持收益序列预测 - 与训练脚本保持一致"""
    def __init__(self, input_dim=6, d_model=192, nhead=12, num_layers=4, dim_feedforward=768, dropout=0.1, label_horizon=10):
        super().__init__()
        self.d_model = d_model
        self.label_horizon = label_horizon
        
        # Encoder - 与训练脚本保持一致
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Decoder - 与训练脚本保持一致
        self.output_embedding = nn.Linear(1, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'  # 使用GELU激活函数
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # 输出层 - 与训练脚本保持一致
        self.fc_hidden = nn.Linear(d_model, d_model // 2)
        self.fc_out = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, src, tgt, teacher_forcing=True):
        batch_size = src.size(0)
        
        # Encoder - 与训练脚本保持一致
        src_emb = self.input_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        memory = self.transformer_encoder(src_emb)
        memory = self.layer_norm(memory)  # 添加层归一化
        
        if teacher_forcing:
            # 训练模式 - 使用teacher forcing
            start_token = torch.zeros((batch_size, 1), device=src.device)
            tgt_in = torch.cat([start_token, tgt[:, :-1]], dim=1)
            
            tgt_emb = self.output_embedding(tgt_in.unsqueeze(-1)) * math.sqrt(self.d_model)
            tgt_emb = self.decoder_pos_encoding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.label_horizon).to(src.device)
            decoder_output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            
            # 改进的输出处理
            hidden = self.fc_hidden(decoder_output)
            hidden = torch.relu(hidden)  # 添加激活函数
            hidden = self.dropout(hidden)
            out = self.fc_out(hidden).squeeze(-1)
        else:
            # 推理模式 - 自回归生成
            output = torch.zeros((batch_size, self.label_horizon), device=src.device)
            current_input = torch.zeros((batch_size, 1), device=src.device)
            
            for i in range(self.label_horizon):
                tgt_emb = self.output_embedding(current_input.unsqueeze(-1)) * math.sqrt(self.d_model)
                tgt_emb = self.decoder_pos_encoding(tgt_emb)
                tgt_emb = self.dropout(tgt_emb)
                
                # 修复掩码生成：对于自回归生成，每次只需要一个时间步的掩码
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(1).to(src.device)
                decoder_output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # 改进的输出处理
                hidden = self.fc_hidden(decoder_output[:, -1:])  # [batch_size, 1, d_model//2]
                hidden = torch.relu(hidden)
                hidden = self.dropout(hidden)
                next_pred = self.fc_out(hidden)  # [batch_size, 1, 1]
                next_pred = next_pred.view(batch_size)  # 明确重塑为 [batch_size]
                
                output[:, i] = next_pred
                current_input = next_pred.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
            
            out = output
        
        return out

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class NewTrainingBacktest:
    """新训练结果回测分析"""
    
    def __init__(self, data_file='direct_features_latest.csv', 
                 stock_data_file='stock_price_vol_d.txt'):
        self.data_file = data_file
        self.stock_data_file = stock_data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = None
    
    def load_data(self):
        """加载数据 - 增加调试信息"""
        print("正在加载数据...")
        
        if not os.path.exists(self.data_file):
            print(f"数据文件不存在: {self.data_file}")
            return None, None
        
        try:
            features_df = pd.read_csv(self.data_file)
            features_df['end_date'] = pd.to_datetime(features_df['end_date'])
            features_df['StockID'] = features_df['StockID'].astype('category')
            print(f"✓ 特征数据加载成功: {len(features_df)}条记录, {features_df['StockID'].nunique()}只股票")
            print(f"  日期范围: {features_df['end_date'].min()} 到 {features_df['end_date'].max()}")
        except Exception as e:
            print(f"特征数据加载失败: {e}")
            return None, None
        
        try:
            stock_data = pd.read_feather(self.stock_data_file)
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data['StockID'] = stock_data['StockID'].astype('category')
            print(f"✓ 价格数据加载成功: {len(stock_data)}条记录, {stock_data['StockID'].nunique()}只股票")
            print(f"  日期范围: {stock_data['date'].min()} 到 {stock_data['date'].max()}")
        except Exception as e:
            print(f"价格数据加载失败: {e}")
            return None, None
        
        return features_df, stock_data
    
    def detect_model_config(self, model_file):
        """检测模型配置"""
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'config' in checkpoint:
                return checkpoint['config']
        except:
            pass
        
        # 与训练脚本保持一致的默认配置
        return {
            'input_dim': 6,
            'd_model': 192,
            'nhead': 12,
            'num_layers': 4,
            'dim_feedforward': 768,
            'dropout': 0.1,
            'label_horizon': 10
        }
    
    def load_models(self):
        """加载训练好的模型"""
        models = []
        model_files = [
            'trained_models/transformer_model_seed_123.pth',
            'trained_models/transformer_model_seed_456.pth',
            'trained_models/transformer_model_seed_789.pth'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                model_config = self.detect_model_config(model_file)
                if self.model_config is None:
                    self.model_config = model_config
                
                checkpoint = torch.load(model_file, map_location=self.device)
                model = TransformerSeq2SeqModel(**model_config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                models.append(model)
        
        return models if models else None
    
    def predict_factor_scores(self, features_df, models, batch_size=4096):
        """批量预测因子得分 - 优化版本"""
        print("正在预测因子得分（优化版本）...")
        
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        seq_len = 40
        
        # 预处理特征序列
        all_feature_seqs = []
        all_dates = []
        all_stocks = []
        
        for stock_id in tqdm(features_df['StockID'].unique(), desc="预处理特征序列"):
            stock_features = features_df[features_df['StockID'] == stock_id]
            if len(stock_features) < seq_len:
                continue
            feature_values = stock_features[feature_cols].values
            dates = stock_features['end_date'].values
            for i in range(len(stock_features) - seq_len + 1):
                all_feature_seqs.append(feature_values[i:i+seq_len])
                all_dates.append(dates[i+seq_len-1])
                all_stocks.append(stock_id)
        
        all_feature_seqs = np.array(all_feature_seqs, dtype=np.float32)
        all_dates_array = np.array(all_dates)
        
        # 按时间排序
        sorted_indices = np.argsort(all_dates_array)
        all_feature_seqs = all_feature_seqs[sorted_indices]
        all_dates = all_dates_array[sorted_indices]
        all_stocks = [all_stocks[i] for i in sorted_indices]
        
        # 时间分割
        split_idx = int(0.8 * len(all_feature_seqs))
        train_features = all_feature_seqs[:split_idx]
        val_features = all_feature_seqs[split_idx:]
        train_dates = all_dates[:split_idx]
        val_dates = all_dates[split_idx:]
        
        # 滚动窗口标准化 - 与训练文件保持一致
        try:
            from ultra_fast_scaler import UltraFastRollingScaler
            rolling_scaler = UltraFastRollingScaler(window_size=252)
            
            # 强制使用真正的滚动窗口标准化
            print(f"数据规模: {train_features.size:,} 元素，强制使用滚动窗口标准化")
            
            train_features_scaled = rolling_scaler.fit_transform(
                train_features, train_dates, use_cache=True, use_simple=False
            )
            val_features_scaled = rolling_scaler.transform(
                val_features, val_dates, use_simple=False
            )
            print("✓ 使用UltraFastRollingScaler进行标准化")
        except ImportError:
            # 简单标准化作为后备
            print("使用简单标准化（建议安装ultra_fast_scaler）")
            train_features_scaled = (train_features - np.mean(train_features, axis=(0,1), keepdims=True)) / (np.std(train_features, axis=(0,1), keepdims=True) + 1e-8)
            val_features_scaled = (val_features - np.mean(train_features, axis=(0,1), keepdims=True)) / (np.std(train_features, axis=(0,1), keepdims=True) + 1e-8)
            train_features_scaled = np.clip(train_features_scaled, -3, 3)  # 特征标准化范围
            val_features_scaled = np.clip(val_features_scaled, -3, 3)
        
        all_feature_seqs = np.concatenate([train_features_scaled, val_features_scaled], axis=0)
        
        # 批量推理 - 优化版本
        all_preds = []
        print(f"开始批量推理，总批次: {len(range(0, len(all_feature_seqs), batch_size))}")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_feature_seqs), batch_size), desc="批量推理"):
                batch_feats = torch.tensor(all_feature_seqs[i:i+batch_size], dtype=torch.float32, device=self.device)
                dummy_target = torch.zeros((batch_feats.shape[0], 10), dtype=torch.float32, device=self.device)
                
                # 集成预测 - 修复序列到序列预测
                batch_ensemble = []
                for model in models:
                    # 关键修复：使用自回归模式进行序列预测
                    # 这里我们需要逐步生成T+1到T+10的收益序列
                    preds = model(batch_feats, dummy_target, teacher_forcing=False)
                    
                    # 检查预测质量
                    if i == 0:  # 只在第一个批次打印统计信息
                        for t in range(min(6, preds.shape[1])):
                            pred_t = preds[:, t]
                            print(f"模型T+{t+1}预测统计: mean={pred_t.mean().item():.6f}, std={pred_t.std().item():.6f}")
                    
                    # 数值稳定性处理
                    if torch.isnan(preds).any() or torch.isinf(preds).any():
                        nan_count = torch.isnan(preds).sum().item()
                        inf_count = torch.isinf(preds).sum().item()
                        if i == 0:
                            print(f"检测到异常值: NaN={nan_count}, Inf={inf_count}")
                        preds = torch.where(torch.isnan(preds) | torch.isinf(preds), torch.zeros_like(preds), preds)
                    
                    # 合理的数值范围（收益率通常在-50%到50%之间）
                    preds = torch.clamp(preds, -0.5, 0.5)
                    batch_ensemble.append(preds.cpu().numpy())
                
                # 集成平均
                batch_ensemble = np.mean(batch_ensemble, axis=0)
                
                # 最终数值检查
                if np.isnan(batch_ensemble).any() or np.isinf(batch_ensemble).any():
                    if i == 0:
                        nan_count = np.isnan(batch_ensemble).sum()
                        inf_count = np.isinf(batch_ensemble).sum()
                        print(f"集成后异常值: NaN={nan_count}, Inf={inf_count}")
                    batch_ensemble = np.nan_to_num(batch_ensemble, nan=0.0, posinf=0.1, neginf=-0.1)
                
                all_preds.append(batch_ensemble)
        
        all_preds = np.concatenate(all_preds, axis=0)
        print(f"✓ 批量推理完成，预测形状: {all_preds.shape}")
        
        # 详细检查预测质量
        pred_stats = {
            'mean': np.mean(all_preds),
            'std': np.std(all_preds),
            'min': np.min(all_preds),
            'max': np.max(all_preds),
            'median': np.median(all_preds),
            'q25': np.percentile(all_preds, 25),
            'q75': np.percentile(all_preds, 75),
            'nan_count': np.isnan(all_preds).sum(),
            'inf_count': np.isinf(all_preds).sum(),
            'zero_count': (all_preds == 0.0).sum(),
            'nonzero_ratio': (all_preds != 0.0).mean()
        }
        print(f"详细预测统计: {pred_stats}")
        
        # 检查每个时间步的预测质量
        for t in range(min(6, all_preds.shape[1])):
            t_preds = all_preds[:, t]
            t_stats = {
                'mean': np.mean(t_preds),
                'std': np.std(t_preds),
                'min': np.min(t_preds),
                'max': np.max(t_preds)
            }
            print(f"T+{t+1}时间步统计: {t_stats}")
        
        # 如果预测方差过小，说明模型没有有效预测能力
        if pred_stats['std'] < 1e-6:
            print("警告：模型预测方差极小，模型可能未能学到有效模式")
            print("建议检查：1)模型训练是否充分 2)数据标准化是否正确 3)损失函数是否合适")
        
        # 创建因子得分DataFrame - 对应T+1到T+6的预测
        factor_dfs = []
        for t in range(6):  # T+1到T+6
            factor_scores_t = all_preds[:, t].copy()
            
            # 基本清理
            factor_scores_t = np.where(np.isnan(factor_scores_t) | np.isinf(factor_scores_t), 0.0, factor_scores_t)
            
            print(f"T+{t+1}原始收益率预测分布: min={np.min(factor_scores_t):.6f}, max={np.max(factor_scores_t):.6f}, std={np.std(factor_scores_t):.6f}")
            
            # 对于收益率预测，我们可以直接使用原始值作为因子得分
            # 因为收益率本身就有明确的方向性含义
            if np.std(factor_scores_t) > 1e-8:
                # 只做轻微的异常值处理
                q01 = np.percentile(factor_scores_t, 1)
                q99 = np.percentile(factor_scores_t, 99)
                factor_scores_t = np.clip(factor_scores_t, q01, q99)
                
                # 可选：如果希望标准化，可以进行中性化处理
                # factor_scores_t = factor_scores_t - np.mean(factor_scores_t)
            else:
                print(f"T+{t+1}预测方差过小，保持原始值")
            
            print(f"T+{t+1}处理后因子分布: min={np.min(factor_scores_t):.6f}, max={np.max(factor_scores_t):.6f}, std={np.std(factor_scores_t):.6f}")
            
            factor_df_t = pd.DataFrame({
                'StockID': all_stocks,
                'date': all_dates,
                'factor_score': factor_scores_t,
                'time_window': t + 1
            })
            factor_dfs.append(factor_df_t)
        
        factor_df = pd.concat(factor_dfs, ignore_index=True)
        print(f"✓ 因子得分DataFrame创建完成: {factor_df.shape}")
        return factor_df
    
    def calculate_returns(self, stock_data, factor_df):
        """计算收益率 - 修复时间窗口处理逻辑"""
        print("正在计算收益率...")
        start_date = pd.to_datetime('2017-01-01')
        end_date = pd.to_datetime('2025-02-28')
        
        # 首先，我们需要为每个股票-日期组合计算所有时间窗口的收益率
        # 而不是根据time_window来匹配
        
        # 获取所有唯一的股票-日期组合
        unique_stock_dates = factor_df[['StockID', 'date']].drop_duplicates()
        print(f"唯一股票-日期组合数: {len(unique_stock_dates)}")
        
        # 合并价格数据
        merged_data = pd.merge(
            unique_stock_dates, 
            stock_data[['StockID', 'StockName', 'date', 'close', 'open', 'high', 'low', 'vol', 'amount']], 
            on=['StockID', 'date'], 
            how='inner'
        )
        
        print(f"合并后数据量: {len(merged_data)}")
        
        # 时间筛选
        merged_data = merged_data[
            (merged_data['date'] >= start_date) & 
            (merged_data['date'] <= end_date)
        ]
        
        print(f"时间筛选后数据量: {len(merged_data)}")
        
        # 应用股票池筛选
        merged_data = self._apply_stock_filters(merged_data)
        print(f"股票池筛选后数据量: {len(merged_data)}")
        
        if len(merged_data) == 0:
            print("警告：筛选后没有有效数据")
            return pd.DataFrame()
        
        # 确保数据类型正确
        merged_data = merged_data.sort_values(['StockID', 'date']).reset_index(drop=True)
        
        # 计算未来收益率
        print("正在计算未来收益率...")
        for window in range(1, 7):
            merged_data[f'future_price_{window}d'] = merged_data.groupby('StockID')['close'].shift(-window)
            merged_data[f'return_{window}d'] = (merged_data[f'future_price_{window}d'] - merged_data['close']) / merged_data['close']
        
        print(f"计算收益率后数据量: {len(merged_data)}")
        
        # 现在将因子得分数据与收益率数据合并
        # 我们需要为每个时间窗口创建对应的记录
        final_results = []
        
        for time_window in range(1, 7):
            # 获取对应时间窗口的因子得分
            factor_window = factor_df[factor_df['time_window'] == time_window].copy()
            print(f"时间窗口{time_window}因子数据量: {len(factor_window)}")
            
            if len(factor_window) == 0:
                print(f"时间窗口{time_window}没有因子数据，跳过")
                continue
            
            # 合并因子得分和收益率
            window_data = pd.merge(
                factor_window[['StockID', 'date', 'factor_score']],
                merged_data[['StockID', 'date', f'return_{time_window}d']],
                on=['StockID', 'date'],
                how='inner'
            )
            
            # 添加时间窗口标识
            window_data['time_window'] = time_window
            
            # 清理无效数据
            window_data = window_data.dropna(subset=['factor_score', f'return_{time_window}d'])
            
            if len(window_data) > 0:
                print(f"时间窗口{time_window}有效数据量: {len(window_data)}")
                final_results.append(window_data)
            else:
                print(f"时间窗口{time_window}清理后无有效数据")
        
        if not final_results:
            print("警告：没有有效的时间窗口数据")
            return pd.DataFrame()
        
        returns_df = pd.concat(final_results, ignore_index=True)
        print(f"✓ 收益率计算完成，最终数据量: {len(returns_df)}")
        print(f"✓ 最终数据列: {returns_df.columns.tolist()}")
        
        # 检查每个时间窗口的数据量
        for tw in range(1, 7):
            tw_data = returns_df[returns_df['time_window'] == tw]
            print(f"时间窗口{tw}最终数据量: {len(tw_data)}")
        
        return returns_df
    
    def _apply_stock_filters(self, merged_data):
        """应用股票池筛选 - 放宽筛选条件"""
        def is_st_stock(stock_name):
            if pd.isna(stock_name):
                return False
            stock_name = str(stock_name).upper()
            # 只过滤明显的ST股票，放宽条件
            st_keywords = ['*ST', 'S*ST', '*SST', 'PT']
            return any(keyword in stock_name for keyword in st_keywords)
        
        def is_suspended(vol, amount):
            if pd.isna(vol) or pd.isna(amount):
                return True
            # 放宽停牌判断：成交量或成交额为0都算停牌
            return vol == 0 or amount == 0
        
        def is_limit_up(open_price, close_price, high_price):
            if pd.isna(open_price) or pd.isna(close_price) or pd.isna(high_price):
                return False
            # 放宽涨停判断：涨幅超过9.5%且接近最高价
            return abs(close_price - high_price) < 0.02 * close_price and (close_price / open_price - 1) > 0.095
        
        # 记录筛选前的数据量
        original_count = len(merged_data)
        
        # 应用筛选
        st_mask = merged_data['StockName'].apply(is_st_stock)
        merged_data = merged_data[~st_mask]
        st_filtered = original_count - len(merged_data)
        
        suspended_mask = merged_data.apply(lambda x: is_suspended(x['vol'], x['amount']), axis=1)
        merged_data = merged_data[~suspended_mask]
        suspended_filtered = original_count - st_filtered - len(merged_data)
        
        limit_up_mask = merged_data.apply(lambda x: is_limit_up(x['open'], x['close'], x['high']), axis=1)
        merged_data = merged_data[~limit_up_mask]
        limit_up_filtered = original_count - st_filtered - suspended_filtered - len(merged_data)
        
        print(f"股票筛选统计: 原始{original_count}只, ST过滤{st_filtered}只, 停牌过滤{suspended_filtered}只, 涨停过滤{limit_up_filtered}只, 剩余{len(merged_data)}只")
        
        return merged_data
    
    def calculate_ic_metrics(self, returns_df):
        """计算IC指标 - 优化版本，降低筛选阈值"""
        print("正在计算IC指标（优化版本）...")
        dates = sorted(returns_df['date'].unique())
        ic_results = []
        
        # 统计信息
        total_dates = len(dates)
        skipped_dates = 0
        zero_ic_counts = {i: 0 for i in range(1, 7)}
        
        for date in tqdm(dates, desc="计算RankIC"):
            date_data = returns_df[returns_df['date'] == date]
            
            # 降低样本数量要求：从30降到15
            if len(date_data) < 15:
                skipped_dates += 1
                continue
            
            ic_results_date = {'date': date}
            
            # 改进的IC计算方法
            t1_to_t6_rank_ics = []
            t1_to_t6_pearson_ics = []
            
            for time_window in range(1, 7):
                window_data = date_data[date_data['time_window'] == time_window]
                
                # 降低样本数量要求：从30降到10
                if len(window_data) < 10:
                    ic_results_date[f'rank_ic_t{time_window}'] = 0
                    ic_results_date[f'pearson_ic_t{time_window}'] = 0
                    zero_ic_counts[time_window] += 1
                    continue
                
                # 关键修复：使用正确的收益率列
                return_col = f'return_{time_window}d'
                
                # 检查是否存在对应的收益率列
                if return_col not in window_data.columns:
                    print(f"警告：时间窗口{time_window}缺少{return_col}列")
                    ic_results_date[f'rank_ic_t{time_window}'] = 0
                    ic_results_date[f'pearson_ic_t{time_window}'] = 0
                    zero_ic_counts[time_window] += 1
                    continue
                
                valid_data = window_data.dropna(subset=['factor_score', return_col])
                # 降低样本数量要求：从30降到10
                if len(valid_data) < 10:
                    ic_results_date[f'rank_ic_t{time_window}'] = 0
                    ic_results_date[f'pearson_ic_t{time_window}'] = 0
                    zero_ic_counts[time_window] += 1
                    continue
                
                # 更温和的异常值处理：从5%-95%改为1%-99%
                factor_q01 = valid_data['factor_score'].quantile(0.01)
                factor_q99 = valid_data['factor_score'].quantile(0.99)
                return_q01 = valid_data[return_col].quantile(0.01)
                return_q99 = valid_data[return_col].quantile(0.99)
                
                clean_data = valid_data[
                    (valid_data['factor_score'] >= factor_q01) & 
                    (valid_data['factor_score'] <= factor_q99) &
                    (valid_data[return_col] >= return_q01) & 
                    (valid_data[return_col] <= return_q99)
                ]
                
                # 降低样本数量要求：从30降到8
                if len(clean_data) < 8:
                    ic_results_date[f'rank_ic_t{time_window}'] = 0
                    ic_results_date[f'pearson_ic_t{time_window}'] = 0
                    zero_ic_counts[time_window] += 1
                    continue
                
                factor_scores = clean_data['factor_score'].values.astype(np.float64)
                returns = clean_data[return_col].values.astype(np.float64)
                
                # 放宽数值稳定性检查：从1e-8改为1e-10
                factor_std = np.std(factor_scores)
                return_std = np.std(returns)
                
                if factor_std < 1e-10 or return_std < 1e-10:
                    ic_results_date[f'rank_ic_t{time_window}'] = 0
                    ic_results_date[f'pearson_ic_t{time_window}'] = 0
                    zero_ic_counts[time_window] += 1
                    if time_window <= 3:  # 只对前3个时间窗口打印调试信息
                        print(f"T+{time_window}在{date}：因子std={factor_std:.6f}, 收益std={return_std:.6f}")
                    continue
                
                # 计算RankIC
                rank_ic = fast_rankic_calculation(factor_scores, returns)
                ic_results_date[f'rank_ic_t{time_window}'] = rank_ic
                t1_to_t6_rank_ics.append(rank_ic)
                
                # 计算Pearson IC（与训练时保持一致）
                pearson_ic = fast_ic_calculation(factor_scores, returns)
                ic_results_date[f'pearson_ic_t{time_window}'] = pearson_ic
                t1_to_t6_pearson_ics.append(pearson_ic)
                
                # 调试信息：对于前3个时间窗口，打印详细统计
                if time_window <= 3 and len(dates) < 5:  # 只在前几个日期打印
                    print(f"T+{time_window}在{date}：样本数={len(clean_data)}, RankIC={rank_ic:.6f}, PearsonIC={pearson_ic:.6f}")
            
            # 计算平均值
            if t1_to_t6_rank_ics:
                ic_results_date['avg_rank_ic_t1_t6'] = np.mean(t1_to_t6_rank_ics)
            else:
                ic_results_date['avg_rank_ic_t1_t6'] = 0
                
            if t1_to_t6_pearson_ics:
                ic_results_date['avg_pearson_ic_t1_t6'] = np.mean(t1_to_t6_pearson_ics)
            else:
                ic_results_date['avg_pearson_ic_t1_t6'] = 0
            
            ic_results_date['stock_count'] = len(clean_data) if 'clean_data' in locals() else 0
            ic_results.append(ic_results_date)
        
        ic_df = pd.DataFrame(ic_results)
        print(f"✓ IC计算完成，共{len(ic_df)}个交易日")
        print(f"跳过日期数: {skipped_dates}/{total_dates} ({skipped_dates/total_dates*100:.1f}%)")
        
        # 输出IC统计信息
        if len(ic_df) > 0:
            avg_rank_ic = ic_df['avg_rank_ic_t1_t6'].mean()
            avg_pearson_ic = ic_df['avg_pearson_ic_t1_t6'].mean()
            print(f"平均RankIC: {avg_rank_ic:.6f}")
            print(f"平均PearsonIC: {avg_pearson_ic:.6f}")
            
            # 打印各时间窗口的非零IC统计
            for t in range(1, 7):
                rank_ic_col = f'rank_ic_t{t}'
                if rank_ic_col in ic_df.columns:
                    non_zero_ics = ic_df[ic_df[rank_ic_col] != 0][rank_ic_col]
                    zero_count = zero_ic_counts[t]
                    if len(non_zero_ics) > 0:
                        print(f"T+{t}非零IC统计: 数量={len(non_zero_ics)}, 均值={non_zero_ics.mean():.6f}, 零值数={zero_count}")
                    else:
                        print(f"T+{t}所有IC都为零 (共{zero_count}个)")
        
        return ic_df
    
    def calculate_decile_portfolio_metrics(self, returns_df):
        """计算分10层组合表现指标 - 降低筛选阈值"""
        portfolio_results = []
        date_groups = returns_df.groupby('date')
        
        # 统计信息
        total_dates = len(date_groups)
        skipped_dates = 0
        
        for date, date_data in tqdm(date_groups, desc="计算分层组合"):
            # 降低样本数量要求：从50降到20
            if len(date_data) < 20:
                skipped_dates += 1
                continue
            
            # 多时间窗口数据处理
            if 'time_window' in date_data.columns:
                valid_time_windows = date_data[date_data['time_window'].isin(range(1, 7))]
                if len(valid_time_windows) == 0:
                    skipped_dates += 1
                    continue
                
                time_window_returns = []
                for tw in range(1, 7):
                    tw_data = valid_time_windows[valid_time_windows['time_window'] == tw]
                    # 降低样本数量要求：从50降到15
                    if len(tw_data) < 15:
                        continue
                    
                    return_col = f'return_{tw}d'
                    if return_col not in tw_data.columns:
                        continue
                    
                    valid_data = tw_data.dropna(subset=['factor_score', return_col])
                    # 降低样本数量要求：从50降到15
                    if len(valid_data) < 15:
                        continue
                    
                    valid_data_copy = valid_data[['StockID', 'factor_score', return_col]].copy()
                    valid_data_copy['return_value'] = valid_data_copy[return_col]
                    time_window_returns.append(valid_data_copy[['StockID', 'factor_score', 'return_value']])
                
                if not time_window_returns:
                    skipped_dates += 1
                    continue
                
                combined_data = pd.concat(time_window_returns, ignore_index=True)
                combined_data['factor_score'] = combined_data['factor_score'].astype(np.float64)
                combined_data['return_value'] = combined_data['return_value'].astype(np.float64)
                
                avg_returns = combined_data.groupby('StockID').agg({
                    'factor_score': 'mean',
                    'return_value': 'mean'
                }).reset_index()
                avg_returns.columns = ['StockID', 'factor_score', 'avg_return_t1_t6']
                valid_data = avg_returns
            else:
                return_cols = [col for col in date_data.columns if col.startswith('return_') and col.endswith('d')]
                if not return_cols:
                    skipped_dates += 1
                    continue
                
                valid_cols = ['factor_score'] + return_cols
                valid_data = date_data.dropna(subset=valid_cols)
                # 降低样本数量要求：从50降到20
                if len(valid_data) < 20:
                    skipped_dates += 1
                    continue
                
                valid_data = valid_data.copy()
                valid_data['avg_return_t1_t6'] = valid_data[return_cols].mean(axis=1)
            
            # 降低样本数量要求：从50降到20
            if len(valid_data) < 20:
                skipped_dates += 1
                continue
            
            # 分10层
            valid_data = valid_data.sort_values('factor_score', ascending=False, ignore_index=True)
            decile_size = len(valid_data) // 10
            
            if decile_size < 1:
                skipped_dates += 1
                continue
            
            decile_returns = {}
            for decile in range(1, 11):
                start_idx = (decile - 1) * decile_size
                if decile == 10:
                    end_idx = len(valid_data)
                else:
                    end_idx = decile * decile_size
                
                decile_stocks = valid_data.iloc[start_idx:end_idx]
                decile_returns[f'decile_{decile}_return'] = decile_stocks['avg_return_t1_t6'].mean()
                decile_returns[f'decile_{decile}_count'] = len(decile_stocks)
            
            top_return = decile_returns['decile_1_return']
            bottom_return = decile_returns['decile_10_return']
            
            portfolio_returns = {
                'date': date,
                'top_return': top_return,
                'bottom_return': bottom_return,
                'long_short_return': top_return - bottom_return,
                'total_stock_count': len(valid_data)
            }
            
            portfolio_returns.update(decile_returns)
            portfolio_results.append(portfolio_returns)
        
        portfolio_df = pd.DataFrame(portfolio_results)
        print(f"✓ 投资组合计算完成，共{len(portfolio_df)}个交易日")
        print(f"跳过日期数: {skipped_dates}/{total_dates} ({skipped_dates/total_dates*100:.1f}%)")
        
        return portfolio_df
    
    def calculate_turnover_rate(self, returns_df, top_ratio=0.1):
        """计算TOP组合双边换手率"""
        dates = sorted(returns_df['date'].unique())
        turnover_rates = []
        
        daily_top_stocks = {}
        date_groups = returns_df.groupby('date')
        
        for date, date_data in date_groups:
            if len(date_data) < 50:
                continue
            
            if 'time_window' in date_data.columns:
                valid_time_windows = date_data[date_data['time_window'].isin(range(1, 7))]
                if len(valid_time_windows) == 0:
                    continue
                
                time_window_returns = []
                for tw in range(1, 7):
                    tw_data = valid_time_windows[valid_time_windows['time_window'] == tw]
                    if len(tw_data) < 50:
                        continue
                    
                    return_col = f'return_{tw}d'
                    if return_col not in tw_data.columns:
                        continue
                    
                    valid_data = tw_data.dropna(subset=['factor_score', return_col])
                    if len(valid_data) < 50:
                        continue
                    
                    valid_data_copy = valid_data[['StockID', 'factor_score', return_col]].copy()
                    valid_data_copy['return_value'] = valid_data_copy[return_col]
                    time_window_returns.append(valid_data_copy[['StockID', 'factor_score', 'return_value']])
                
                if not time_window_returns:
                    continue
                
                combined_data = pd.concat(time_window_returns, ignore_index=True)
                combined_data['factor_score'] = combined_data['factor_score'].astype(np.float64)
                combined_data['return_value'] = combined_data['return_value'].astype(np.float64)
                
                avg_returns = combined_data.groupby('StockID').agg({
                    'factor_score': 'mean',
                    'return_value': 'mean'
                }).reset_index()
                valid_data = avg_returns
            else:
                return_cols = [col for col in date_data.columns if col.startswith('return_') and col.endswith('d')]
                if not return_cols:
                    continue
                
                valid_cols = ['factor_score'] + return_cols
                valid_data = date_data.dropna(subset=valid_cols)
                if len(valid_data) < 50:
                    continue
                
                valid_data = valid_data.copy()
                valid_data['avg_return_t1_t6'] = valid_data[return_cols].mean(axis=1)
            
            if len(valid_data) < 50:
                continue
            
            valid_data = valid_data.sort_values('factor_score', ascending=False, ignore_index=True)
            top_count = max(1, int(len(valid_data) * top_ratio))
            top_stocks = set(valid_data.head(top_count)['StockID'])
            daily_top_stocks[date] = top_stocks
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            if prev_date not in daily_top_stocks or curr_date not in daily_top_stocks:
                continue
            
            prev_top_stocks = daily_top_stocks[prev_date]
            curr_top_stocks = daily_top_stocks[curr_date]
            
            if len(prev_top_stocks) == 0 or len(curr_top_stocks) == 0:
                continue
            
            buy_turnover = len(curr_top_stocks - prev_top_stocks) / len(curr_top_stocks)
            sell_turnover = len(prev_top_stocks - curr_top_stocks) / len(prev_top_stocks)
            total_turnover = buy_turnover + sell_turnover
            
            turnover_rates.append({
                'date': curr_date,
                'buy_turnover': buy_turnover,
                'sell_turnover': sell_turnover,
                'total_turnover': total_turnover
            })
        
        turnover_df = pd.DataFrame(turnover_rates)
        return turnover_df
    
    def generate_report(self, ic_df, returns_df, portfolio_df=None, turnover_df=None):
        """生成回测报告"""
        report = {}
        
        report['total_stocks'] = returns_df['StockID'].nunique()
        report['total_observations'] = len(returns_df)
        report['date_range_start'] = returns_df['date'].min().strftime('%Y-%m-%d')
        report['date_range_end'] = returns_df['date'].max().strftime('%Y-%m-%d')
        
        # IC指标
        if 'avg_rank_ic_t1_t6' in ic_df.columns:
            avg_rank_ic_values = ic_df['avg_rank_ic_t1_t6'].dropna()
            if len(avg_rank_ic_values) > 0:
                report['rank_ic_mean'] = avg_rank_ic_values.mean()
                report['rank_ic_std'] = avg_rank_ic_values.std()
                report['rank_ic_ir'] = avg_rank_ic_values.mean() / avg_rank_ic_values.std() if avg_rank_ic_values.std() > 0 else 0
                report['rank_ic_positive_ratio'] = (avg_rank_ic_values > 0).mean()
            else:
                report['rank_ic_mean'] = 0
                report['rank_ic_std'] = 0
                report['rank_ic_ir'] = 0
                report['rank_ic_positive_ratio'] = 0
        
        # 各时间窗口RankIC
        for t in range(1, 7):
            rank_ic_col = f'rank_ic_t{t}'
            if rank_ic_col in ic_df.columns:
                rank_ic_values = ic_df[rank_ic_col].dropna()
                if len(rank_ic_values) > 0:
                    report[f'{rank_ic_col}_mean'] = rank_ic_values.mean()
                    report[f'{rank_ic_col}_std'] = rank_ic_values.std()
                    report[f'{rank_ic_col}_ir'] = rank_ic_values.mean() / rank_ic_values.std() if rank_ic_values.std() > 0 else 0
                    report[f'{rank_ic_col}_positive_ratio'] = (rank_ic_values > 0).mean()
        
        # 组合表现
        if portfolio_df is not None and len(portfolio_df) > 0:
            top_returns = portfolio_df['top_return'].dropna()
            if len(top_returns) > 0:
                report['top_annual_return'] = top_returns.mean() * 52
                report['top_information_ratio'] = top_returns.mean() / top_returns.std() * np.sqrt(52) if top_returns.std() > 0 else 0
                report['top_win_rate'] = (top_returns > 0).mean()
            
            long_short_returns = portfolio_df['long_short_return'].dropna()
            if len(long_short_returns) > 0:
                report['long_short_annual_return'] = long_short_returns.mean() * 52
                report['long_short_information_ratio'] = long_short_returns.mean() / long_short_returns.std() * np.sqrt(52) if long_short_returns.std() > 0 else 0
                report['long_short_win_rate'] = (long_short_returns > 0).mean()
        
        # 换手率
        if turnover_df is not None and len(turnover_df) > 0:
            if 'buy_turnover' in turnover_df.columns:
                report['avg_buy_turnover'] = turnover_df['buy_turnover'].mean()
                report['avg_sell_turnover'] = turnover_df['sell_turnover'].mean()
                report['avg_total_turnover'] = turnover_df['total_turnover'].mean()
                report['turnover_std'] = turnover_df['total_turnover'].std()
        
        # 保存报告
        os.makedirs('factor_results', exist_ok=True)
        with open('factor_results/enhanced_backtest_report.json', 'w') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        ic_df.to_csv('factor_results/enhanced_ic_data.csv', index=False)
        returns_df.to_csv('factor_results/enhanced_returns_data.csv', index=False)
        if portfolio_df is not None:
            portfolio_df.to_csv('factor_results/top_portfolio_data.csv', index=False)
        if turnover_df is not None:
            turnover_df.to_csv('factor_results/turnover_data.csv', index=False)
        
        # 打印报告
        self._print_report(report)
        return report
    
    def _print_report(self, report):
        """打印报告"""
        print("\n" + "="*80)
        print("Transformer因子分析报告")
        print("="*80)
        
        print(f"\n股票池统计:")
        print(f"  股票数量: {report.get('total_stocks', 0):,}")
        print(f"  总观测数: {report.get('total_observations', 0):,}")
        print(f"  时间范围: {report.get('date_range_start', 'N/A')} ~ {report.get('date_range_end', 'N/A')}")
        
        print(f"\n关键指标:")
        print(f"  RankIC均值: {report.get('rank_ic_mean', 0):.6f}")
        print(f"  RankIC标准差: {report.get('rank_ic_std', 0):.6f}")
        print(f"  RankIC IR: {report.get('rank_ic_ir', 0):.6f}")
        print(f"  RankIC>0占比: {report.get('rank_ic_positive_ratio', 0):.2%}")
        print(f"  TOP组合年化收益率: {report.get('top_annual_return', 0):.2%}")
        print(f"  TOP组合信息比率: {report.get('top_information_ratio', 0):.6f}")
        print(f"  TOP组合胜率: {report.get('top_win_rate', 0):.2%}")
        print(f"  TOP组合换手率: {report.get('avg_total_turnover', 0):.2%}")
        
        print(f"\n各时间窗口RankIC:")
        for t in range(1, 7):
            rank_ic_mean = report.get(f'rank_ic_t{t}_mean', 0)
            rank_ic_std = report.get(f'rank_ic_t{t}_std', 0)
            rank_ic_ir = report.get(f'rank_ic_t{t}_ir', 0)
            rank_ic_positive = report.get(f'rank_ic_t{t}_positive_ratio', 0)
            print(f"  T+{t}: RankIC={rank_ic_mean:.6f}, 标准差={rank_ic_std:.6f}, IR={rank_ic_ir:.6f}, >0占比={rank_ic_positive:.2%}")
    
    def plot_results(self, ic_df):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('新训练结果回测分析', fontsize=16, fontweight='bold')
        
        if 'avg_rank_ic_t1_t6' in ic_df.columns:
            # T+1~T+6平均RankIC时间序列
            ax1 = axes[0, 0]
            ax1.plot(ic_df['date'], ic_df['avg_rank_ic_t1_t6'], label='T+1~T+6平均RankIC', alpha=0.7, color='blue')
            ax1.set_title('T+1~T+6平均RankIC时间序列')
            ax1.set_xlabel('日期')
            ax1.set_ylabel('RankIC值')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # T+1 RankIC时间序列
            ax2 = axes[0, 1]
            if 'rank_ic_t1' in ic_df.columns:
                ax2.plot(ic_df['date'], ic_df['rank_ic_t1'], label='T+1 RankIC', alpha=0.7, color='red')
                ax2.set_title('T+1 RankIC时间序列')
            else:
                ax2.text(0.5, 0.5, '没有T+1 RankIC数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('T+1 RankIC时间序列')
            ax2.set_xlabel('日期')
            ax2.set_ylabel('RankIC值')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # T+1~T+6平均RankIC分布直方图
            ax3 = axes[1, 0]
            ax3.hist(ic_df['avg_rank_ic_t1_t6'].dropna(), bins=30, alpha=0.6, label='T+1~T+6平均RankIC', color='blue')
            ax3.set_title('T+1~T+6平均RankIC分布直方图')
            ax3.set_xlabel('RankIC值')
            ax3.set_ylabel('频次')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 股票数量时间序列
        ax4 = axes[1, 1]
        if 'stock_count' in ic_df.columns:
            ax4.plot(ic_df['date'], ic_df['stock_count'], color='green')
            ax4.set_title('每日股票数量')
            ax4.set_xlabel('日期')
            ax4.set_ylabel('股票数量')
        else:
            ax4.text(0.5, 0.5, '没有股票数量数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('股票数量')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('factor_results/new_training_backtest_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_backtest(self):
        """运行回测分析"""
        print("="*80)
        print("新训练结果回测分析")
        print("="*80)
        
        # 1. 加载数据
        features_df, stock_data = self.load_data()
        if features_df is None:
            return
        
        # 2. 加载模型
        models = self.load_models()
        if models is None:
            return
        
        # 3. 预测因子得分
        factor_df = self.predict_factor_scores(features_df, models)
        
        # 4. 计算收益率
        returns_df = self.calculate_returns(stock_data, factor_df)
        
        # 5. 周频调仓处理
        returns_df['week'] = returns_df['date'].dt.to_period('W')
        weekly_dates = returns_df.groupby('week')['date'].max().reset_index()
        weekly_dates = weekly_dates.rename(columns={'date': 'rebalance_date'})
        weekly_returns = pd.merge(returns_df, weekly_dates, on='week', how='inner')
        weekly_returns = weekly_returns[weekly_returns['date'] == weekly_returns['rebalance_date']]
        weekly_returns = weekly_returns.drop(['week', 'rebalance_date'], axis=1)
        
        # 6. 计算IC指标
        ic_df = self.calculate_ic_metrics(weekly_returns)
        
        # 7. 计算分层组合表现
        portfolio_df = self.calculate_decile_portfolio_metrics(weekly_returns)
        
        # 8. 计算换手率
        turnover_df = self.calculate_turnover_rate(weekly_returns)
        
        # 9. 生成报告
        report = self.generate_report(ic_df, weekly_returns, portfolio_df, turnover_df)
        
        # 10. 绘制图表
        self.plot_results(ic_df)
        
        print("\n" + "="*80)
        print("回测分析完成！")
        print("="*80)

def main():
    """主函数"""
    backtest = NewTrainingBacktest()
    backtest.run_backtest()

if __name__ == "__main__":
    main() 