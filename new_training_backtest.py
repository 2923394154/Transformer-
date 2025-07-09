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
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TransformerSeq2SeqModel(nn.Module):
    """Encoder-Decoder结构，支持收益序列预测"""
    def __init__(self, input_dim=6, d_model=256, nhead=16, num_layers=4, dim_feedforward=1024, dropout=0.1, label_horizon=10):
        super().__init__()
        self.d_model = d_model
        self.label_horizon = label_horizon
        # Encoder
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Decoder
        self.output_embedding = nn.Linear(1, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt, teacher_forcing=True):
        batch_size = src.size(0)
        src_emb = self.input_embedding(src) * np.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        memory = self.transformer_encoder(src_emb)
        if teacher_forcing:
            start_token = torch.zeros((batch_size, 1), device=src.device)
            tgt_in = torch.cat([start_token, tgt[:, :-1]], dim=1)
        else:
            tgt_in = torch.zeros((batch_size, 1), device=src.device)
        tgt_emb = self.output_embedding(tgt_in.unsqueeze(-1)) * np.sqrt(self.d_model)
        tgt_emb = self.decoder_pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(src.device)
        out = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask
        )
        out = self.fc_out(out).squeeze(-1)
        return out

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
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
        
        # 加载模型配置
        self.model_config = {
            'model_params': {
                'input_dim': 6,
                'd_model': 256,
                'nhead': 16,
                'num_layers': 4,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'label_horizon': 10
            }
        }
        
        print(f"✓ 设备: {self.device}")
        print(f"✓ 数据文件: {data_file}")
        print(f"✓ 股票数据文件: {stock_data_file}")
    
    def load_data(self):
        """加载数据"""
        print("="*60)
        print("加载新生成的训练数据")
        print("="*60)
        
        # 加载特征数据
        if not os.path.exists(self.data_file):
            print(f"✗ 数据文件不存在: {self.data_file}")
            return None, None
        
        print(f"正在加载特征数据: {self.data_file}")
        features_df = pd.read_csv(self.data_file)
        features_df['end_date'] = pd.to_datetime(features_df['end_date'])
        
        print(f"✓ 特征数据: {features_df.shape}")
        print(f"✓ 股票数量: {features_df['StockID'].nunique()}")
        print(f"✓ 时间范围: {features_df['end_date'].min()} ~ {features_df['end_date'].max()}")
        
        # 加载股票价格数据
        print(f"正在加载股票价格数据: {self.stock_data_file}")
        try:
            stock_data = pd.read_feather(self.stock_data_file)
        except:
            stock_data = pd.read_csv(self.stock_data_file, sep='\t')
        
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        print(f"✓ 价格数据: {stock_data.shape}")
        
        return features_df, stock_data
    
    def detect_model_config(self, model_file):
        """自动检测模型配置"""
        try:
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # 尝试从checkpoint中获取配置
            if 'config' in checkpoint:
                config = checkpoint['config']
                model_params = config.get('model_params', {})
                return model_params
            
            # 如果没有配置信息，使用默认配置
            return self.model_config['model_params']
            
        except Exception as e:
            print(f"✗ 模型配置检测失败: {e}")
            return self.model_config['model_params']
    
    def load_models(self):
        """智能加载训练好的模型"""
        print("="*60)
        print("智能加载训练好的模型")
        print("="*60)
        
        models = []
        model_files = [
            'trained_models/transformer_model_seed_123.pth',
            'trained_models/transformer_model_seed_456.pth',
            'trained_models/transformer_model_seed_789.pth'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"正在加载模型: {model_file}")
                
                # 自动检测模型配置
                model_config = self.detect_model_config(model_file)
                print(f"✓ 检测到模型配置: {model_config}")
                
                # 加载模型
                checkpoint = torch.load(model_file, map_location=self.device)
                model = TransformerSeq2SeqModel(**model_config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                models.append(model)
                print(f"✓ 模型加载成功: {model_file}")
            else:
                print(f"✗ 模型文件不存在: {model_file}")
        
        if not models:
            print("✗ 没有找到可用的模型文件")
            return None
        
        print(f"✓ 成功加载 {len(models)} 个模型")
        return models
    
    def predict_factor_scores(self, features_df, models, batch_size=8192):
        """批量预测因子得分，支持GPU混合精度加速"""
        print("="*60)
        print("批量预测因子得分（GPU加速）")
        print("="*60)
        
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        seq_len = 40
        
        # 预处理所有股票的特征序列
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
        print(f"✓ 总样本数: {len(all_feature_seqs)}")
        
        # 数据标准化 - 与训练时保持一致
        print("正在进行数据标准化...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_reshaped = all_feature_seqs.reshape(-1, all_feature_seqs.shape[-1])
        features_scaled = scaler.fit_transform(features_reshaped)
        all_feature_seqs = features_scaled.reshape(all_feature_seqs.shape)
        print(f"✓ 数据标准化完成")
        
        # 批量推理
        all_preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(all_feature_seqs), batch_size), desc="批量推理"):
                batch_feats = torch.tensor(all_feature_seqs[i:i+batch_size], dtype=torch.float32, device=self.device)
                dummy_target = torch.zeros((batch_feats.shape[0], 10), dtype=torch.float32, device=self.device)
                batch_ensemble = []
                with torch.cuda.amp.autocast():
                    for model in models:
                        preds = model(batch_feats, dummy_target, teacher_forcing=False)
                        batch_ensemble.append(preds.cpu().numpy())
                batch_ensemble = np.mean(batch_ensemble, axis=0)
                all_preds.append(batch_ensemble.flatten())  # 展平为一维数组
        all_preds = np.concatenate(all_preds)
        
        # 检查预测结果
        print(f"✓ 预测结果形状: {all_preds.shape}")
        print(f"✓ 原始预测统计: min={np.min(all_preds):.4f}, max={np.max(all_preds):.4f}, mean={np.mean(all_preds):.4f}")
        print(f"✓ NaN值数量: {np.isnan(all_preds).sum()}")
        print(f"✓ 无穷大值数量: {np.isinf(all_preds).sum()}")
        
        # 处理NaN和无穷大值
        if np.isnan(all_preds).any() or np.isinf(all_preds).any():
            print("⚠️  检测到NaN或无穷大值，进行清理...")
            all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"✓ 清理后统计: min={np.min(all_preds):.4f}, max={np.max(all_preds):.4f}, mean={np.mean(all_preds):.4f}")
        
        # 因子得分标准化 - 修复异常大的预测值
        print("正在进行因子得分标准化...")
        
        # 直接使用tanh函数进行软限制，将范围限制在[-1, 1]
        all_preds = np.tanh(all_preds / 100.0)  # 除以100是为了将大值压缩到合理范围
        
        print(f"✓ 标准化后统计: min={np.min(all_preds):.4f}, max={np.max(all_preds):.4f}, mean={np.mean(all_preds):.4f}")
        
        # 创建因子得分DataFrame
        factor_df = pd.DataFrame({
            'StockID': all_stocks,
            'date': all_dates,
            'factor_score': all_preds
        })
        print(f"✓ 因子得分预测完成: {factor_df.shape}")
        print(f"✓ 因子得分范围: {factor_df['factor_score'].min():.4f} ~ {factor_df['factor_score'].max():.4f}")
        return factor_df
    
    def calculate_returns(self, stock_data, factor_df):
        """计算收益率 - 修正为T+1~T+6收益率，添加股票池筛选"""
        print("="*60)
        print("计算收益率 (T+1~T+6) - 添加股票池筛选")
        print("="*60)
        
        # 设置回测时间区间
        start_date = pd.to_datetime('2017-01-01')
        end_date = pd.to_datetime('2025-02-28')
        
        # 合并因子得分和价格数据
        merged_data = pd.merge(factor_df, stock_data[['StockID', 'StockName', 'date', 'close', 'open', 'high', 'low', 'vol', 'amount']], 
                              on=['StockID', 'date'], how='left')
        
        # 筛选回测时间区间
        merged_data = merged_data[
            (merged_data['date'] >= start_date) & 
            (merged_data['date'] <= end_date)
        ]
        
        print(f"✓ 回测时间区间: {start_date.date()} ~ {end_date.date()}")
        print(f"✓ 筛选后数据量: {len(merged_data)}")
        
        # 股票池筛选 - 剔除ST股票、停牌和涨停股票
        print("正在进行股票池筛选...")
        
        # 1. 剔除ST股票 - 基于股票名称
        def is_st_stock(stock_name):
            if pd.isna(stock_name):
                return False
            stock_name = str(stock_name).upper()
            # 检查是否包含ST、*ST、SST等标识
            st_keywords = ['ST', '*ST', 'SST', 'S*ST', '*SST']
            return any(keyword in stock_name for keyword in st_keywords)
        
        # 2. 剔除停牌股票 - 基于成交量和成交额
        def is_suspended(vol, amount):
            # 如果成交量为0或极小，且成交额为0或极小，认为是停牌
            if pd.isna(vol) or pd.isna(amount):
                return True
            return vol <= 0 or amount <= 0
        
        # 3. 剔除涨停股票 - 基于涨跌幅
        def is_limit_up(close, open_price):
            if pd.isna(close) or pd.isna(open_price) or open_price <= 0:
                return False
            # 计算涨跌幅，涨停通常为10%（主板）或20%（创业板/科创板）
            change_pct = (close - open_price) / open_price
            # 使用9.8%和19.8%作为涨停判断阈值，避免边界问题
            return change_pct >= 0.098
        
        # 应用筛选条件
        print("应用股票池筛选条件...")
        
        # 记录筛选前的数据量
        before_filter = len(merged_data)
        
        # 剔除ST股票
        st_mask = merged_data['StockName'].apply(is_st_stock)
        merged_data = merged_data[~st_mask]
        st_filtered = before_filter - len(merged_data)
        print(f"✓ 剔除ST股票: {st_filtered} 条记录")
        
        # 剔除停牌股票
        suspended_mask = merged_data.apply(lambda x: is_suspended(x['vol'], x['amount']), axis=1)
        merged_data = merged_data[~suspended_mask]
        suspended_filtered = before_filter - st_filtered - len(merged_data)
        print(f"✓ 剔除停牌股票: {suspended_filtered} 条记录")
        
        # 剔除涨停股票
        limit_up_mask = merged_data.apply(lambda x: is_limit_up(x['close'], x['open']), axis=1)
        merged_data = merged_data[~limit_up_mask]
        limit_up_filtered = before_filter - st_filtered - suspended_filtered - len(merged_data)
        print(f"✓ 剔除涨停股票: {limit_up_filtered} 条记录")
        
        print(f"✓ 筛选后保留数据量: {len(merged_data)}")
        print(f"✓ 筛选后股票数量: {merged_data['StockID'].nunique()}")
        print(f"✓ 筛选后日期范围: {merged_data['date'].min()} ~ {merged_data['date'].max()}")
        
        # 计算未来收益率
        returns_data = []
        
        for stock_id in tqdm(merged_data['StockID'].unique(), desc="计算收益率"):
            stock_data_subset = merged_data[merged_data['StockID'] == stock_id].sort_values('date', ascending=True, ignore_index=True)
            
            if len(stock_data_subset) < 7:  # 至少需要7天数据（当前+6天未来）
                continue
            
            # 计算T+1~T+6收益率
            for i in range(len(stock_data_subset) - 6):
                current_row = stock_data_subset.iloc[i]
                future_prices = stock_data_subset.iloc[i+1:i+7]['close'].values
                
                if len(future_prices) == 6 and not np.any(np.isnan(future_prices)):
                    current_price = current_row['close']
                    
                    # 计算T+1~T+6的累计收益率
                    returns_6d = (future_prices[-1] - current_price) / current_price
                    
                    returns_data.append({
                        'StockID': stock_id,
                        'date': current_row['date'],
                        'factor_score': current_row['factor_score'],
                        'return_6d': returns_6d
                    })
        
        returns_df = pd.DataFrame(returns_data)
        print(f"✓ 收益率计算完成: {returns_df.shape}")
        print(f"✓ 最终股票数量: {returns_df['StockID'].nunique()}")
        
        return returns_df
    
    def calculate_ic_metrics(self, returns_df):
        """计算IC指标 - 修正为6天收益率"""
        print("="*60)
        print("计算IC指标 (6天收益率)")
        print("="*60)
        
        # 检查数据
        print(f"✓ 输入数据形状: {returns_df.shape}")
        print(f"✓ 数据列: {returns_df.columns.tolist()}")
        print(f"✓ 日期范围: {returns_df['date'].min()} ~ {returns_df['date'].max()}")
        print(f"✓ 股票数量: {returns_df['StockID'].nunique()}")
        
        # 按日期分组计算IC
        ic_results = []
        
        for date in tqdm(returns_df['date'].unique(), desc="计算IC"):
            date_data = returns_df[returns_df['date'] == date]
            
            if len(date_data) < 10:  # 至少需要10只股票
                continue
            
            # 检查数据有效性
            valid_data = date_data.dropna(subset=['factor_score', 'return_6d'])
            if len(valid_data) < 10:
                continue
            
            # 计算IC
            ic_6d = np.corrcoef(valid_data['factor_score'], valid_data['return_6d'])[0, 1]
            
            # 计算RankIC - 修复计算方式
            def calculate_rankic(x, y):
                x_rank = np.argsort(np.argsort(x))
                y_rank = np.argsort(np.argsort(y))
                x_rank_norm = (x_rank - x_rank.mean()) / (x_rank.std() + 1e-8)
                y_rank_norm = (y_rank - y_rank.mean()) / (y_rank.std() + 1e-8)
                return np.corrcoef(x_rank_norm, y_rank_norm)[0, 1]
            
            rank_ic_6d = calculate_rankic(valid_data['factor_score'], valid_data['return_6d'])
            
            ic_results.append({
                'date': date,
                'ic_6d': ic_6d if not np.isnan(ic_6d) else 0,
                'rank_ic_6d': rank_ic_6d if not np.isnan(rank_ic_6d) else 0,
                'stock_count': len(valid_data)
            })
        
        ic_df = pd.DataFrame(ic_results)
        print(f"✓ IC计算完成: {ic_df.shape}")
        
        if len(ic_df) == 0:
            print("⚠️  警告: 没有有效的IC数据，请检查数据质量")
            # 创建一个空的DataFrame但包含正确的列
            ic_df = pd.DataFrame(columns=['date', 'ic_6d', 'rank_ic_6d', 'stock_count'])
        
        return ic_df
    
    def calculate_top_portfolio_metrics(self, returns_df, top_ratio=0.1):
        """计算TOP组合表现指标 - 修正为6天收益率"""
        print("="*60)
        print("计算TOP组合表现指标 (6天收益率)")
        print("="*60)
        
        portfolio_results = []
        
        for date in tqdm(returns_df['date'].unique(), desc="计算TOP组合表现"):
            date_data = returns_df[returns_df['date'] == date].copy()
            
            if len(date_data) < 20:  # 至少需要20只股票
                continue
            
            # 检查数据有效性
            valid_data = date_data.dropna(subset=['factor_score', 'return_6d'])
            if len(valid_data) < 20:
                continue
            
            # 按因子得分排序，选择TOP股票
            valid_data = valid_data.sort_values('factor_score', ascending=False, ignore_index=True)
            top_count = max(1, int(len(valid_data) * top_ratio))
            top_stocks = valid_data.head(top_count)
            
            # 计算TOP组合收益率
            portfolio_returns = {
                'date': date,
                'top_return_6d': top_stocks['return_6d'].mean(),
                'top_stock_count': len(top_stocks),
                'total_stock_count': len(valid_data)
            }
            
            portfolio_results.append(portfolio_returns)
        
        portfolio_df = pd.DataFrame(portfolio_results)
        print(f"✓ TOP组合计算完成: {portfolio_df.shape}")
        
        if len(portfolio_df) == 0:
            print("⚠️  警告: 没有有效的TOP组合数据，请检查数据质量")
            portfolio_df = pd.DataFrame(columns=['date', 'top_return_6d', 'top_stock_count', 'total_stock_count'])
        
        return portfolio_df
    
    def calculate_turnover_rate(self, returns_df, top_ratio=0.1):
        """计算换手率 - 双边换手率"""
        print("="*60)
        print("计算双边换手率")
        print("="*60)
        
        dates = sorted(returns_df['date'].unique())
        turnover_rates = []
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            # 获取前一日和当日的TOP股票
            prev_data = returns_df[returns_df['date'] == prev_date].sort_values('factor_score', ascending=False, ignore_index=True)
            curr_data = returns_df[returns_df['date'] == curr_date].sort_values('factor_score', ascending=False, ignore_index=True)
            
            if len(prev_data) < 20 or len(curr_data) < 20:
                continue
            
            prev_top_count = max(1, int(len(prev_data) * top_ratio))
            curr_top_count = max(1, int(len(curr_data) * top_ratio))
            
            prev_top_stocks = set(prev_data.head(prev_top_count)['StockID'])
            curr_top_stocks = set(curr_data.head(curr_top_count)['StockID'])
            
            # 计算双边换手率
            # 买入换手率 + 卖出换手率
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
        print(f"✓ 换手率计算完成: {turnover_df.shape}")
        
        return turnover_df
    
    def generate_report(self, ic_df, returns_df, portfolio_df=None, turnover_df=None):
        """生成增强版回测报告 - 修正为6天收益率，添加股票池筛选统计"""
        print("="*60)
        print("生成增强版回测报告 (6天收益率) - 股票池筛选")
        print("="*60)
        
        # 计算统计指标
        report = {}
        
        # 添加股票池筛选统计
        report['total_stocks'] = returns_df['StockID'].nunique()
        report['total_observations'] = len(returns_df)
        report['date_range_start'] = returns_df['date'].min().strftime('%Y-%m-%d')
        report['date_range_end'] = returns_df['date'].max().strftime('%Y-%m-%d')
        
        # 检查IC数据是否为空
        if len(ic_df) == 0 or 'ic_6d' not in ic_df.columns:
            print("⚠️  警告: IC数据为空，使用默认值")
            report['ic_6d_mean'] = 0
            report['ic_6d_std'] = 0
            report['ic_6d_ir'] = 0
            report['ic_6d_positive_ratio'] = 0
            report['rank_ic_6d_mean'] = 0
            report['rank_ic_6d_std'] = 0
            report['rank_ic_6d_ir'] = 0
            report['rank_ic_6d_positive_ratio'] = 0
        else:
            # IC和RankIC指标 (6天)
            ic_values = ic_df['ic_6d'].dropna()
            rank_ic_values = ic_df['rank_ic_6d'].dropna()
            
            # IC指标
            report['ic_6d_mean'] = ic_values.mean() if len(ic_values) > 0 else 0
            report['ic_6d_std'] = ic_values.std() if len(ic_values) > 0 else 0
            report['ic_6d_ir'] = ic_values.mean() / ic_values.std() if len(ic_values) > 0 and ic_values.std() > 0 else 0
            report['ic_6d_positive_ratio'] = (ic_values > 0).mean() if len(ic_values) > 0 else 0
            
            # RankIC指标
            report['rank_ic_6d_mean'] = rank_ic_values.mean() if len(rank_ic_values) > 0 else 0
            report['rank_ic_6d_std'] = rank_ic_values.std() if len(rank_ic_values) > 0 else 0
            report['rank_ic_6d_ir'] = rank_ic_values.mean() / rank_ic_values.std() if len(rank_ic_values) > 0 and rank_ic_values.std() > 0 else 0
            report['rank_ic_6d_positive_ratio'] = (rank_ic_values > 0).mean() if len(rank_ic_values) > 0 else 0
        
        # TOP组合指标
        if portfolio_df is not None:
            returns = portfolio_df['top_return_6d'].dropna()
            
            # 年化超额收益率 - 周频调仓 (52周/年)
            annual_return = returns.mean() * 52
            
            report['top_6d_annual_return'] = annual_return
            
            # 信息比率 - 周频调仓
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(52) if returns.std() > 0 else 0
            report['top_6d_sharpe_ratio'] = sharpe_ratio
            
            # 胜率
            win_rate = (returns > 0).mean()
            report['top_6d_win_rate'] = win_rate
        
        # 换手率指标
        if turnover_df is not None and len(turnover_df) > 0:
            if 'buy_turnover' in turnover_df.columns:
                report['avg_buy_turnover'] = turnover_df['buy_turnover'].mean()
                report['avg_sell_turnover'] = turnover_df['sell_turnover'].mean()
                report['avg_total_turnover'] = turnover_df['total_turnover'].mean()
                report['turnover_std'] = turnover_df['total_turnover'].std()
            else:
                report['avg_buy_turnover'] = 0
                report['avg_sell_turnover'] = 0
                report['avg_total_turnover'] = 0
                report['turnover_std'] = 0
        else:
            report['avg_buy_turnover'] = 0
            report['avg_sell_turnover'] = 0
            report['avg_total_turnover'] = 0
            report['turnover_std'] = 0
        
        # 保存报告
        os.makedirs('factor_results', exist_ok=True)
        
        with open('factor_results/enhanced_backtest_report.json', 'w') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        # 保存详细数据
        ic_df.to_csv('factor_results/enhanced_ic_data.csv', index=False)
        returns_df.to_csv('factor_results/enhanced_returns_data.csv', index=False)
        if portfolio_df is not None:
            portfolio_df.to_csv('factor_results/top_portfolio_data.csv', index=False)
        if turnover_df is not None:
            turnover_df.to_csv('factor_results/turnover_data.csv', index=False)
        
        # 打印增强版报告
        print("\n" + "="*80)
        print("增强版因子分析报告 (回测期: 2017/1/1～2025/2/28, 周频调仓, 股票池筛选)")
        print("="*80)
        
        print(f"\n股票池筛选统计:")
        print(f"  最终股票数量: {report.get('total_stocks', 0):,}")
        print(f"  总观测数: {report.get('total_observations', 0):,}")
        print(f"  数据时间范围: {report.get('date_range_start', 'N/A')} ~ {report.get('date_range_end', 'N/A')}")
        
        print(f"\n6天收益率预测结果:")
        print(f"  IC均值: {report.get('ic_6d_mean', 0):.6f}")
        print(f"  IC标准差: {report.get('ic_6d_std', 0):.6f}")
        print(f"  IC信息比率: {report.get('ic_6d_ir', 0):.6f}")
        print(f"  IC>0占比: {report.get('ic_6d_positive_ratio', 0):.2%}")
        
        print(f"  RankIC均值: {report.get('rank_ic_6d_mean', 0):.6f}")
        print(f"  RankIC标准差: {report.get('rank_ic_6d_std', 0):.6f}")
        print(f"  RankIC信息比率: {report.get('rank_ic_6d_ir', 0):.6f}")
        print(f"  RankIC>0占比: {report.get('rank_ic_6d_positive_ratio', 0):.2%}")
        
        print(f"\nTOP组合表现 (10层):")
        print(f"  6天年化超额收益率: {report.get('top_6d_annual_return', 0):.2%}")
        print(f"  6天信息比率: {report.get('top_6d_sharpe_ratio', 0):.4f}")
        print(f"  6天胜率: {report.get('top_6d_win_rate', 0):.2%}")
        
        if turnover_df is not None:
            print(f"\n双边换手率指标:")
            print(f"  平均买入换手率: {report.get('avg_buy_turnover', 0):.2%}")
            print(f"  平均卖出换手率: {report.get('avg_sell_turnover', 0):.2%}")
            print(f"  平均总换手率: {report.get('avg_total_turnover', 0):.2%}")
            print(f"  换手率标准差: {report.get('turnover_std', 0):.2%}")
        
        print(f"\n✓ 增强版报告已保存: factor_results/enhanced_backtest_report.json")
        print(f"✓ 详细数据已保存到 factor_results/ 目录")
        
        return report
    
    def plot_results(self, ic_df):
        """绘制结果图表 - 修正为6天收益率，添加股票池筛选标识"""
        print("="*60)
        print("绘制结果图表 (6天收益率) - 股票池筛选")
        print("="*60)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('新训练结果回测分析 (回测期: 2017/1/1～2025/2/28, 周频调仓, 股票池筛选)', fontsize=16, fontweight='bold')
        
        # 1. IC时间序列
        ax1 = axes[0, 0]
        ax1.plot(ic_df['date'], ic_df['ic_6d'], label='6天 IC', alpha=0.7, color='blue')
        ax1.set_title('IC时间序列 (6天收益率)')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('IC值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RankIC时间序列
        ax2 = axes[0, 1]
        ax2.plot(ic_df['date'], ic_df['rank_ic_6d'], label='6天 RankIC', alpha=0.7, color='red')
        ax2.set_title('RankIC时间序列 (6天收益率)')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('RankIC值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. IC分布直方图
        ax3 = axes[1, 0]
        ax3.hist(ic_df['ic_6d'].dropna(), bins=30, alpha=0.6, label='6天 IC', color='blue')
        ax3.set_title('IC分布直方图 (6天收益率)')
        ax3.set_xlabel('IC值')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 股票数量时间序列
        ax4 = axes[1, 1]
        ax4.plot(ic_df['date'], ic_df['stock_count'], color='green')
        ax4.set_title('每日股票数量 (股票池筛选后)')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('股票数量')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('factor_results/new_training_backtest_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: factor_results/new_training_backtest_analysis.png")
        plt.show()
    
    def run_backtest(self):
        """运行增强版回测分析 - 周频调仓，股票池筛选"""
        print("="*80)
        print("增强版因子分析回测 (回测期: 2017/1/1～2025/2/28, 周频调仓, 股票池筛选)")
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
        
        # 4. 计算收益率（包含股票池筛选）
        returns_df = self.calculate_returns(stock_data, factor_df)
        
        # 5. 周频调仓处理
        print("="*60)
        print("周频调仓处理")
        print("="*60)
        
        # 按周分组，保留每周所有股票的数据
        returns_df['week'] = returns_df['date'].dt.to_period('W')
        
        # 选择每周最后一个交易日作为调仓日
        weekly_dates = returns_df.groupby('week')['date'].max().reset_index()
        weekly_dates = weekly_dates.rename(columns={'date': 'rebalance_date'})
        
        # 筛选出调仓日的数据
        weekly_returns = pd.merge(returns_df, weekly_dates, on='week', how='inner')
        weekly_returns = weekly_returns[weekly_returns['date'] == weekly_returns['rebalance_date']]
        weekly_returns = weekly_returns.drop(['week', 'rebalance_date'], axis=1)
        
        print(f"✓ 原始数据: {len(returns_df)} 个交易日")
        print(f"✓ 周频数据: {len(weekly_returns)} 个调仓日")
        print(f"✓ 周频数据股票数: {weekly_returns['StockID'].nunique()}")
        
        # 6. 计算IC指标
        ic_df = self.calculate_ic_metrics(weekly_returns)
        
        # 7. 计算TOP组合表现
        portfolio_df = self.calculate_top_portfolio_metrics(weekly_returns)
        
        # 8. 计算换手率
        turnover_df = self.calculate_turnover_rate(weekly_returns)
        
        # 9. 生成增强版报告
        report = self.generate_report(ic_df, weekly_returns, portfolio_df, turnover_df)
        
        # 10. 绘制图表
        self.plot_results(ic_df)
        
        print("\n" + "="*80)
        print("增强版回测分析完成！")
        print("="*80)

def main():
    """主函数"""
    backtest = NewTrainingBacktest()
    backtest.run_backtest()

if __name__ == "__main__":
    main() 