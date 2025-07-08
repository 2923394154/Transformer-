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
    
    def load_models(self):
        """加载训练好的模型"""
        print("="*60)
        print("加载训练好的模型")
        print("="*60)
        
        models = []
        model_files = [
            'trained_models/transformer_model_seed_42.pth',
            'trained_models/transformer_model_seed_142.pth',
            'trained_models/transformer_model_seed_242.pth'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"正在加载模型: {model_file}")
                
                # 加载模型
                checkpoint = torch.load(model_file, map_location=self.device)
                model = TransformerSeq2SeqModel(**self.model_config['model_params'])
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
    
    def predict_factor_scores(self, features_df, models, batch_size=4096):
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
                all_preds.append(batch_ensemble[:, 0])  # 只取第一天预测
        all_preds = np.concatenate(all_preds)
        
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
        """计算收益率"""
        print("="*60)
        print("计算收益率")
        print("="*60)
        
        # 合并因子得分和价格数据
        merged_data = pd.merge(factor_df, stock_data[['StockID', 'date', 'close']], 
                              on=['StockID', 'date'], how='left')
        
        # 计算未来收益率
        returns_data = []
        
        for stock_id in tqdm(merged_data['StockID'].unique(), desc="计算收益率"):
            stock_data_subset = merged_data[merged_data['StockID'] == stock_id].sort_values('date')
            
            if len(stock_data_subset) < 2:
                continue
            
            # 计算未来1天、5天、10天收益率
            for i in range(len(stock_data_subset) - 10):
                current_row = stock_data_subset.iloc[i]
                future_prices = stock_data_subset.iloc[i+1:i+11]['close'].values
                
                if len(future_prices) == 10 and not np.any(np.isnan(future_prices)):
                    current_price = current_row['close']
                    returns_1d = (future_prices[0] - current_price) / current_price
                    returns_5d = (future_prices[4] - current_price) / current_price
                    returns_10d = (future_prices[9] - current_price) / current_price
                    
                    returns_data.append({
                        'StockID': stock_id,
                        'date': current_row['date'],
                        'factor_score': current_row['factor_score'],
                        'return_1d': returns_1d,
                        'return_5d': returns_5d,
                        'return_10d': returns_10d
                    })
        
        returns_df = pd.DataFrame(returns_data)
        print(f"✓ 收益率计算完成: {returns_df.shape}")
        
        return returns_df
    
    def calculate_ic_metrics(self, returns_df):
        """计算IC指标"""
        print("="*60)
        print("计算IC指标")
        print("="*60)
        
        # 按日期分组计算IC
        ic_results = []
        
        for date in tqdm(returns_df['date'].unique(), desc="计算IC"):
            date_data = returns_df[returns_df['date'] == date]
            
            if len(date_data) < 10:  # 至少需要10只股票
                continue
            
            # 计算IC
            ic_1d = np.corrcoef(date_data['factor_score'], date_data['return_1d'])[0, 1]
            ic_5d = np.corrcoef(date_data['factor_score'], date_data['return_5d'])[0, 1]
            ic_10d = np.corrcoef(date_data['factor_score'], date_data['return_10d'])[0, 1]
            
            # 计算RankIC
            rank_ic_1d = spearmanr(date_data['factor_score'], date_data['return_1d'])[0]
            rank_ic_5d = spearmanr(date_data['factor_score'], date_data['return_5d'])[0]
            rank_ic_10d = spearmanr(date_data['factor_score'], date_data['return_10d'])[0]
            
            ic_results.append({
                'date': date,
                'ic_1d': ic_1d if not np.isnan(ic_1d) else 0,
                'ic_5d': ic_5d if not np.isnan(ic_5d) else 0,
                'ic_10d': ic_10d if not np.isnan(ic_10d) else 0,
                'rank_ic_1d': rank_ic_1d if not np.isnan(rank_ic_1d) else 0,
                'rank_ic_5d': rank_ic_5d if not np.isnan(rank_ic_5d) else 0,
                'rank_ic_10d': rank_ic_10d if not np.isnan(rank_ic_10d) else 0,
                'stock_count': len(date_data)
            })
        
        ic_df = pd.DataFrame(ic_results)
        print(f"✓ IC计算完成: {ic_df.shape}")
        
        return ic_df
    
    def generate_report(self, ic_df, returns_df):
        """生成回测报告"""
        print("="*60)
        print("生成回测报告")
        print("="*60)
        
        # 计算统计指标
        report = {}
        
        for period in ['1d', '5d', '10d']:
            ic_col = f'ic_{period}'
            rank_ic_col = f'rank_ic_{period}'
            
            if ic_col in ic_df.columns:
                ic_values = ic_df[ic_col].dropna()
                rank_ic_values = ic_df[rank_ic_col].dropna()
                
                report[f'{period}_ic_mean'] = ic_values.mean()
                report[f'{period}_ic_std'] = ic_values.std()
                report[f'{period}_ic_ir'] = ic_values.mean() / ic_values.std() if ic_values.std() > 0 else 0
                report[f'{period}_ic_positive_ratio'] = (ic_values > 0).mean()
                
                report[f'{period}_rank_ic_mean'] = rank_ic_values.mean()
                report[f'{period}_rank_ic_std'] = rank_ic_values.std()
                report[f'{period}_rank_ic_ir'] = rank_ic_values.mean() / rank_ic_values.std() if rank_ic_values.std() > 0 else 0
                report[f'{period}_rank_ic_positive_ratio'] = (rank_ic_values > 0).mean()
        
        # 保存报告
        os.makedirs('factor_results', exist_ok=True)
        
        with open('factor_results/new_training_backtest_report.json', 'w') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        # 保存详细数据
        ic_df.to_csv('factor_results/new_training_ic_data.csv', index=False)
        returns_df.to_csv('factor_results/new_training_returns_data.csv', index=False)
        
        # 打印报告
        print("\n" + "="*80)
        print("新训练结果回测报告")
        print("="*80)
        
        for period in ['1d', '5d', '10d']:
            print(f"\n{period}期预测结果:")
            print(f"  IC均值: {report.get(f'{period}_ic_mean', 0):.6f}")
            print(f"  IC标准差: {report.get(f'{period}_ic_std', 0):.6f}")
            print(f"  IC信息比率: {report.get(f'{period}_ic_ir', 0):.6f}")
            print(f"  IC>0占比: {report.get(f'{period}_ic_positive_ratio', 0):.2%}")
            print(f"  RankIC均值: {report.get(f'{period}_rank_ic_mean', 0):.6f}")
            print(f"  RankIC标准差: {report.get(f'{period}_rank_ic_std', 0):.6f}")
            print(f"  RankIC信息比率: {report.get(f'{period}_rank_ic_ir', 0):.6f}")
            print(f"  RankIC>0占比: {report.get(f'{period}_rank_ic_positive_ratio', 0):.2%}")
        
        print(f"\n✓ 报告已保存: factor_results/new_training_backtest_report.json")
        print(f"✓ IC数据已保存: factor_results/new_training_ic_data.csv")
        print(f"✓ 收益率数据已保存: factor_results/new_training_returns_data.csv")
        
        return report
    
    def plot_results(self, ic_df):
        """绘制结果图表"""
        print("="*60)
        print("绘制结果图表")
        print("="*60)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('新训练结果回测分析', fontsize=16, fontweight='bold')
        
        # 1. IC时间序列
        ax1 = axes[0, 0]
        for period in ['1d', '5d', '10d']:
            ic_col = f'ic_{period}'
            if ic_col in ic_df.columns:
                ax1.plot(ic_df['date'], ic_df[ic_col], label=f'{period} IC', alpha=0.7)
        ax1.set_title('IC时间序列')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('IC值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RankIC时间序列
        ax2 = axes[0, 1]
        for period in ['1d', '5d', '10d']:
            rank_ic_col = f'rank_ic_{period}'
            if rank_ic_col in ic_df.columns:
                ax2.plot(ic_df['date'], ic_df[rank_ic_col], label=f'{period} RankIC', alpha=0.7)
        ax2.set_title('RankIC时间序列')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('RankIC值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. IC分布直方图
        ax3 = axes[1, 0]
        for period in ['1d', '5d', '10d']:
            ic_col = f'ic_{period}'
            if ic_col in ic_df.columns:
                ax3.hist(ic_df[ic_col].dropna(), bins=30, alpha=0.6, label=f'{period} IC')
        ax3.set_title('IC分布直方图')
        ax3.set_xlabel('IC值')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 股票数量时间序列
        ax4 = axes[1, 1]
        ax4.plot(ic_df['date'], ic_df['stock_count'])
        ax4.set_title('每日股票数量')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('股票数量')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('factor_results/new_training_backtest_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: factor_results/new_training_backtest_analysis.png")
        plt.show()
    
    def run_backtest(self):
        """运行完整回测"""
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
        
        # 5. 计算IC指标
        ic_df = self.calculate_ic_metrics(returns_df)
        
        # 6. 生成报告
        report = self.generate_report(ic_df, returns_df)
        
        # 7. 绘制图表
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