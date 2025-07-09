#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单个因子模型分析脚本
分析单个模型的因子表现，支持选择不同的随机种子模型
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

class SingleFactorAnalyzer:
    """单个因子模型分析器"""
    
    def __init__(self, data_file='direct_features_latest.csv', 
                 stock_data_file='stock_price_vol_d.txt'):
        self.data_file = data_file
        self.stock_data_file = stock_data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU优化设置
        if torch.cuda.is_available():
            # 设置GPU内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
            torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试
            torch.backends.cudnn.deterministic = False  # 关闭确定性计算以提高速度
            
            # 显示GPU信息
            print(f"✓ GPU: {torch.cuda.get_device_name()}")
            print(f"✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        else:
            print("⚠️  未检测到GPU，将使用CPU进行计算")
        
        print(f"✓ 设备: {self.device}")
        print(f"✓ 数据文件: {data_file}")
        print(f"✓ 股票数据文件: {stock_data_file}")
    
    def monitor_gpu_memory(self):
        """监控GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU内存使用: {allocated:.2f}GB / {cached:.2f}GB / {total:.2f}GB")
    
    def load_data(self):
        """加载数据"""
        print("="*60)
        print("加载数据")
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
    
    def load_model(self, model_file):
        """加载模型"""
        if not os.path.exists(model_file):
            print(f"✗ 模型文件不存在: {model_file}")
            return None
        
        print(f"正在加载模型: {model_file}")
        
        # 定义简化的模型类
        class PositionalEncoding(nn.Module):
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
        
        class TransformerSeq2SeqModel(nn.Module):
            def __init__(self, input_dim=6, d_model=256, nhead=16, num_layers=4, dim_feedforward=1024, dropout=0.1, label_horizon=10):
                super().__init__()
                self.d_model = d_model
                self.label_horizon = label_horizon
                # Encoder
                self.input_embedding = nn.Linear(input_dim, d_model)
                self.pos_encoding = PositionalEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True, norm_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                # Decoder
                self.output_embedding = nn.Linear(1, d_model)
                self.decoder_pos_encoding = PositionalEncoding(d_model)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True, norm_first=True
                )
                self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
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
                out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                out = self.fc_out(out).squeeze(-1)
                return out
        
        # 使用默认配置
        model_config = {
            'input_dim': 6,
            'd_model': 256,
            'nhead': 16,
            'num_layers': 4,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'label_horizon': 10
        }
        
        checkpoint = torch.load(model_file, map_location=self.device)
        model = TransformerSeq2SeqModel(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✓ 模型加载成功: {model_file}")
        return model
    
    def predict_factor_scores(self, features_df, model, batch_size=8192):
        """预测因子得分"""
        print("="*60)
        print("预测因子得分")
        print("="*60)
        
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
        print(f"✓ 总样本数: {len(all_feature_seqs)}")
        
        # 数据标准化
        print("正在进行数据标准化...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_reshaped = all_feature_seqs.reshape(-1, all_feature_seqs.shape[-1])
        features_scaled = scaler.fit_transform(features_reshaped)
        all_feature_seqs = features_scaled.reshape(all_feature_seqs.shape)
        print(f"✓ 数据标准化完成")
        
        # 启用混合精度推理
        print("启用混合精度推理...")
        from torch.cuda.amp import autocast
        
        # 批量推理
        all_preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(all_feature_seqs), batch_size), desc="批量推理"):
                batch_feats = torch.tensor(all_feature_seqs[i:i+batch_size], dtype=torch.float32, device=self.device)
                dummy_target = torch.zeros((batch_feats.shape[0], 10), dtype=torch.float32, device=self.device)
                
                # 使用混合精度推理
                with autocast():
                    preds = model(batch_feats, dummy_target, teacher_forcing=False)
                    all_preds.append(preds.cpu().numpy().flatten())
                
                # 清理GPU内存
                del batch_feats, dummy_target, preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 每10个批次监控一次GPU内存
                if (i // batch_size) % 10 == 0:
                    self.monitor_gpu_memory()
        
        all_preds = np.concatenate(all_preds)
        
        # 检查和处理预测结果
        print(f"✓ 预测结果形状: {all_preds.shape}")
        print(f"✓ 预测结果统计: min={np.min(all_preds):.4f}, max={np.max(all_preds):.4f}, mean={np.mean(all_preds):.4f}")
        
        if np.isnan(all_preds).any() or np.isinf(all_preds).any():
            print("⚠️  检测到NaN或无穷大值，进行清理...")
            all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 创建因子得分DataFrame
        factor_df = pd.DataFrame({
            'StockID': all_stocks,
            'date': all_dates,
            'factor_score': all_preds
        })
        print(f"✓ 因子得分预测完成: {factor_df.shape}")
        return factor_df
    
    def calculate_returns_and_ic(self, factor_df, stock_data):
        """计算收益率和IC指标"""
        print("="*60)
        print("计算收益率和IC指标")
        print("="*60)
        
        # 设置回测时间范围
        start_date = pd.to_datetime('2017-01-01')
        end_date = pd.to_datetime('2025-02-28')
        
        # 过滤数据到指定时间范围
        factor_df = factor_df[(factor_df['date'] >= start_date) & (factor_df['date'] <= end_date)].copy()
        stock_data = stock_data[(stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)].copy()
        
        print(f"✓ 回测期间: {start_date.date()} ~ {end_date.date()}")
        print(f"✓ 因子数据: {factor_df.shape}")
        print(f"✓ 价格数据: {stock_data.shape}")
        
        # 合并数据
        merged_data = pd.merge(factor_df, stock_data[['StockID', 'date', 'close']], 
                              on=['StockID', 'date'], how='left')
        
        # 计算收益率 - T+1到T+6（5个交易日）
        returns_data = []
        for stock_id in tqdm(merged_data['StockID'].unique(), desc="计算收益率"):
            stock_data_subset = merged_data[merged_data['StockID'] == stock_id].copy()
            stock_data_subset = stock_data_subset.sort_values('date')
            
            if len(stock_data_subset) < 7:  # 至少需要7天数据
                continue
            
            for i in range(len(stock_data_subset) - 6):
                current_row = stock_data_subset.iloc[i]
                future_prices = stock_data_subset.iloc[i+1:i+7]['close'].values
                
                if len(future_prices) >= 5 and not np.any(np.isnan(future_prices)):
                    current_price = current_row['close']
                    
                    # T+1到T+6收益率（5个交易日）
                    returns_5d = (future_prices[4] - current_price) / current_price
                    
                    returns_data.append({
                        'StockID': stock_id,
                        'date': current_row['date'],
                        'factor_score': current_row['factor_score'],
                        'return_5d': returns_5d
                    })
        
        returns_df = pd.DataFrame(returns_data)
        print(f"✓ 收益率计算完成: {returns_df.shape}")
        
        # 计算IC指标
        ic_results = []
        for date in tqdm(returns_df['date'].unique(), desc="计算IC"):
            date_data = returns_df[returns_df['date'] == date]
            
            if len(date_data) < 10:  # 至少需要10只股票
                continue
            
            # 计算RankIC
            rank_ic_5d = spearmanr(date_data['factor_score'], date_data['return_5d'])[0]
            
            ic_results.append({
                'date': date,
                'rank_ic_5d': rank_ic_5d if not np.isnan(rank_ic_5d) else 0,
                'stock_count': len(date_data)
            })
        
        ic_df = pd.DataFrame(ic_results)
        print(f"✓ IC计算完成: {ic_df.shape}")
        
        return returns_df, ic_df
    
    def calculate_portfolio_metrics(self, returns_df, top_ratio=0.1):
        """计算组合表现指标 - 分10层测试"""
        print("="*60)
        print("计算组合表现指标 - 分10层测试")
        print("="*60)
        
        # 分10层测试
        num_deciles = 10
        decile_ratio = 1.0 / num_deciles
        
        # TOP组合表现（第10层）
        portfolio_results = []
        for date in tqdm(returns_df['date'].unique(), desc="计算TOP组合表现"):
            date_data = returns_df[returns_df['date'] == date].copy()
            
            if len(date_data) < 20:  # 至少需要20只股票
                continue
            
            # 按因子得分排序，分10层
            date_data = date_data.sort_values('factor_score', ascending=False)
            total_stocks = len(date_data)
            
            # 第10层（TOP层）
            top_start = int(0)  # 第1层
            top_end = int(total_stocks * decile_ratio)  # 前10%
            top_stocks = date_data.iloc[top_start:top_end]
            
            # 第1层（BOTTOM层）
            bottom_start = int(total_stocks * (1 - decile_ratio))  # 后10%
            bottom_end = total_stocks
            bottom_stocks = date_data.iloc[bottom_start:bottom_end]
            
            portfolio_results.append({
                'date': date,
                'top_return_5d': top_stocks['return_5d'].mean(),
                'bottom_return_5d': bottom_stocks['return_5d'].mean(),
                'long_short_return_5d': top_stocks['return_5d'].mean() - bottom_stocks['return_5d'].mean(),
                'top_stock_count': len(top_stocks),
                'bottom_stock_count': len(bottom_stocks),
                'total_stock_count': total_stocks
            })
        
        portfolio_df = pd.DataFrame(portfolio_results)
        print(f"✓ TOP组合计算完成: {portfolio_df.shape}")
        
        # 双边换手率计算
        dates = sorted(returns_df['date'].unique())
        turnover_rates = []
        
        for i in range(1, len(dates)):
            prev_date = dates[i-1]
            curr_date = dates[i]
            
            prev_data = returns_df[returns_df['date'] == prev_date].copy()
            curr_data = returns_df[returns_df['date'] == curr_date].copy()
            
            if len(prev_data) < 20 or len(curr_data) < 20:
                continue
            
            prev_data = prev_data.sort_values('factor_score', ascending=False)
            curr_data = curr_data.sort_values('factor_score', ascending=False)
            
            # 计算TOP层换手率
            prev_total = len(prev_data)
            curr_total = len(curr_data)
            
            prev_top_count = max(1, int(prev_total * decile_ratio))
            curr_top_count = max(1, int(curr_total * decile_ratio))
            
            prev_top_stocks = set(prev_data.head(prev_top_count)['StockID'])
            curr_top_stocks = set(curr_data.head(curr_top_count)['StockID'])
            
            # 双边换手率 = (卖出数量 + 买入数量) / (期初持仓 + 期末持仓)
            sell_count = len(prev_top_stocks - curr_top_stocks)  # 卖出
            buy_count = len(curr_top_stocks - prev_top_stocks)    # 买入
            total_holdings = len(prev_top_stocks) + len(curr_top_stocks)
            
            if total_holdings > 0:
                turnover = (sell_count + buy_count) / total_holdings
            else:
                turnover = 0.0
            
            turnover_rates.append({
                'date': curr_date,
                'turnover_rate': turnover,
                'sell_count': sell_count,
                'buy_count': buy_count,
                'total_holdings': total_holdings
            })
        
        turnover_df = pd.DataFrame(turnover_rates)
        print(f"✓ 双边换手率计算完成: {turnover_df.shape}")
        
        return portfolio_df, turnover_df
    
    def generate_report(self, ic_df, returns_df, portfolio_df, turnover_df, model_name):
        """生成分析报告"""
        print("="*60)
        print(f"生成{model_name}分析报告")
        print("="*60)
        
        # 计算统计指标
        report = {}
        
        # 5日RankIC指标
        rank_ic_values = ic_df['rank_ic_5d'].dropna()
        report['rank_ic_mean'] = rank_ic_values.mean()
        report['rank_ic_std'] = rank_ic_values.std()
        report['rank_ic_ir'] = rank_ic_values.mean() / rank_ic_values.std() if rank_ic_values.std() > 0 else 0
        report['rank_ic_positive_ratio'] = (rank_ic_values > 0).mean()
        
        # TOP组合指标（第10层）
        top_returns = portfolio_df['top_return_5d'].dropna()
        bottom_returns = portfolio_df['bottom_return_5d'].dropna()
        long_short_returns = portfolio_df['long_short_return_5d'].dropna()
        
        # 年化收益率（周频调仓，一年约52周）
        report['top_annual_return'] = top_returns.mean() * 52
        report['bottom_annual_return'] = bottom_returns.mean() * 52
        report['long_short_annual_return'] = long_short_returns.mean() * 52
        
        # 信息比率
        report['top_sharpe_ratio'] = top_returns.mean() / top_returns.std() * np.sqrt(52) if top_returns.std() > 0 else 0
        report['bottom_sharpe_ratio'] = bottom_returns.mean() / bottom_returns.std() * np.sqrt(52) if bottom_returns.std() > 0 else 0
        report['long_short_sharpe_ratio'] = long_short_returns.mean() / long_short_returns.std() * np.sqrt(52) if long_short_returns.std() > 0 else 0
        
        # 胜率
        report['top_win_rate'] = (top_returns > 0).mean()
        report['bottom_win_rate'] = (bottom_returns > 0).mean()
        report['long_short_win_rate'] = (long_short_returns > 0).mean()
        
        # 双边换手率指标
        report['avg_turnover_rate'] = turnover_df['turnover_rate'].mean()
        report['turnover_std'] = turnover_df['turnover_rate'].std()
        report['avg_sell_count'] = turnover_df['sell_count'].mean()
        report['avg_buy_count'] = turnover_df['buy_count'].mean()
        
        # 保存报告
        os.makedirs('factor_results', exist_ok=True)
        
        report_file = f'factor_results/{model_name}_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        # 保存详细数据
        ic_df.to_csv(f'factor_results/{model_name}_ic_data.csv', index=False)
        returns_df.to_csv(f'factor_results/{model_name}_returns_data.csv', index=False)
        portfolio_df.to_csv(f'factor_results/{model_name}_portfolio_data.csv', index=False)
        turnover_df.to_csv(f'factor_results/{model_name}_turnover_data.csv', index=False)
        
        # 打印报告
        print(f"\n{'='*80}")
        print(f"{model_name}因子分析报告")
        print(f"{'='*80}")
        print(f"回测期间: 2017/1/1 ~ 2025/2/28")
        print(f"调仓频率: 周频")
        print(f"收益率计算: T+1~T+6（5个交易日）")
        print(f"分层测试: 10层")
        print(f"换手率: 双边换手率")
        
        print(f"\nRankIC指标:")
        print(f"  RankIC均值: {report.get('rank_ic_mean', 0):.6f}")
        print(f"  RankIC标准差: {report.get('rank_ic_std', 0):.6f}")
        print(f"  RankIC信息比率: {report.get('rank_ic_ir', 0):.6f}")
        print(f"  RankIC>0占比: {report.get('rank_ic_positive_ratio', 0):.2%}")
        
        print(f"\n10层测试表现:")
        print(f"  TOP层（第10层）年化收益率: {report.get('top_annual_return', 0):.2%}")
        print(f"  TOP层信息比率: {report.get('top_sharpe_ratio', 0):.4f}")
        print(f"  TOP层胜率: {report.get('top_win_rate', 0):.2%}")
        
        print(f"  BOTTOM层（第1层）年化收益率: {report.get('bottom_annual_return', 0):.2%}")
        print(f"  BOTTOM层信息比率: {report.get('bottom_sharpe_ratio', 0):.4f}")
        print(f"  BOTTOM层胜率: {report.get('bottom_win_rate', 0):.2%}")
        
        print(f"  多空组合年化收益率: {report.get('long_short_annual_return', 0):.2%}")
        print(f"  多空组合信息比率: {report.get('long_short_sharpe_ratio', 0):.4f}")
        print(f"  多空组合胜率: {report.get('long_short_win_rate', 0):.2%}")
        
        print(f"\n双边换手率指标:")
        print(f"  平均换手率: {report.get('avg_turnover_rate', 0):.2%}")
        print(f"  换手率标准差: {report.get('turnover_std', 0):.2%}")
        print(f"  平均卖出数量: {report.get('avg_sell_count', 0):.1f}")
        print(f"  平均买入数量: {report.get('avg_buy_count', 0):.1f}")
        
        print(f"\n✓ 报告已保存: {report_file}")
        print(f"✓ 详细数据已保存到 factor_results/ 目录")
        
        return report
    
    def analyze_model(self, model_seed):
        """分析指定种子的模型"""
        model_file = f'trained_models/transformer_model_seed_{model_seed}.pth'
        model_name = f'model_seed_{model_seed}'
        
        print("="*80)
        print(f"分析模型: {model_name}")
        print("="*80)
        
        # 1. 加载数据
        features_df, stock_data = self.load_data()
        if features_df is None:
            return None
        
        # 2. 加载模型
        model = self.load_model(model_file)
        if model is None:
            return None
        
        # 3. 预测因子得分
        factor_df = self.predict_factor_scores(features_df, model)
        
        # 4. 计算收益率和IC指标
        returns_df, ic_df = self.calculate_returns_and_ic(factor_df, stock_data)
        
        # 5. 计算组合表现指标
        portfolio_df, turnover_df = self.calculate_portfolio_metrics(returns_df)
        
        # 6. 生成报告
        report = self.generate_report(ic_df, returns_df, portfolio_df, turnover_df, model_name)
        
        return report

def main():
    """主函数"""
    # 可用的模型种子
    available_seeds = [123, 456, 789]
    
    print("可用的模型种子:")
    for i, seed in enumerate(available_seeds, 1):
        print(f"  {i}. {seed}")
    
    # 用户选择
    while True:
        try:
            choice = input(f"\n请选择要分析的模型种子 (1-{len(available_seeds)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(available_seeds):
                selected_seed = available_seeds[choice_idx]
                break
            else:
                print("无效选择，请重新输入")
        except ValueError:
            print("请输入有效的数字")
    
    print(f"\n开始分析模型种子: {selected_seed}")
    
    # 创建分析器并运行分析
    analyzer = SingleFactorAnalyzer()
    report = analyzer.analyze_model(selected_seed)
    
    if report:
        print(f"\n✓ 分析完成！模型种子 {selected_seed} 的分析报告已生成")
    else:
        print(f"\n✗ 分析失败！请检查模型文件是否存在")

if __name__ == "__main__":
    main() 