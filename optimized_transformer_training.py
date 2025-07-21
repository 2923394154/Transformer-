#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化速度的Transformer因子训练
主要优化：
1. 平衡模型容量和速度
2. 优化批次大小
3. 简化损失函数
4. 改进数据加载
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import math
from datetime import datetime
import os
import json
import psutil
import argparse
warnings.filterwarnings('ignore')

# GPU加速设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

class FactorDataset(Dataset):
    """收益序列预测数据集"""
    def __init__(self, features, labels, seq_len=40, label_horizon=10):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len
        self.label_horizon = label_horizon
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature_seq = self.features[idx]
        label_seq = self.labels[idx]
        return feature_seq, label_seq

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FastTransformerModel(nn.Module):
    """快速Transformer模型 - 平衡速度和效果"""
    def __init__(self, input_dim=6, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, dropout=0.1, label_horizon=10):
        super().__init__()
        self.d_model = d_model
        self.label_horizon = label_horizon
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='relu'  # 使用ReLU加速
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 解码器
        self.output_embedding = nn.Linear(1, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, src, tgt, teacher_forcing=True):
        batch_size = src.size(0)
        
        # 编码器
        src_emb = self.input_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.input_norm(src_emb)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        memory = self.transformer_encoder(src_emb)
        
        if teacher_forcing:
            # 训练模式
            start_token = torch.zeros((batch_size, 1), device=src.device)
            tgt_in = torch.cat([start_token, tgt[:, :-1]], dim=1)
            
            tgt_emb = self.output_embedding(tgt_in.unsqueeze(-1)) * math.sqrt(self.d_model)
            tgt_emb = self.output_norm(tgt_emb)
            tgt_emb = self.decoder_pos_encoding(tgt_emb)
            tgt_emb = self.dropout(tgt_emb)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.label_horizon).to(src.device)
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            out = self.fc_out(out).squeeze(-1)
        else:
            # 推理模式
            output = torch.zeros((batch_size, self.label_horizon), device=src.device)
            
            for i in range(self.label_horizon):
                if i == 0:
                    current_input = torch.zeros((batch_size, 1), device=src.device)
                else:
                    current_input = output[:, :i]
                
                tgt_emb = self.output_embedding(current_input.unsqueeze(-1)) * math.sqrt(self.d_model)
                tgt_emb = self.output_norm(tgt_emb)
                tgt_emb = self.decoder_pos_encoding(tgt_emb)
                tgt_emb = self.dropout(tgt_emb)
                
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_input.size(1)).to(src.device)
                decoder_out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                next_output = self.fc_out(decoder_out[:, -1:, :]).squeeze(-1)
                
                if torch.isnan(next_output).any() or torch.isinf(next_output).any():
                    next_output = torch.where(torch.isnan(next_output) | torch.isinf(next_output), 
                                            torch.zeros_like(next_output), next_output)
                
                output[:, i:i+1] = next_output
            
            out = output
        
        # 数值稳定性
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.where(torch.isnan(out) | torch.isinf(out), torch.zeros_like(out), out)
        
        return out

def fast_ic_loss(predictions, targets):
    """快速IC损失函数"""
    # 输入检查
    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    if torch.isnan(targets).any() or torch.isinf(targets).any():
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    # 简化的IC计算 - 只计算T+1到T+3
    ic_list = []
    
    for t in range(min(3, predictions.shape[1])):  # 只计算前3个时间步
        pred = predictions[:, t]
        true = targets[:, t]
        
        # 快速过滤
        valid_mask = ~(torch.isnan(pred) | torch.isnan(true) | torch.isinf(pred) | torch.isinf(true))
        if valid_mask.sum() < 10:
            continue
        
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        
        # 快速标准化
        pred_mean = pred_valid.mean()
        pred_std = pred_valid.std()
        true_mean = true_valid.mean()
        true_std = true_valid.std()
        
        if pred_std < 1e-8 or true_std < 1e-8:
            continue
        
        pred_norm = (pred_valid - pred_mean) / torch.clamp(pred_std, min=1e-8)
        true_norm = (true_valid - true_mean) / torch.clamp(true_std, min=1e-8)
        
        covariance = torch.mean(pred_norm * true_norm)
        ic = torch.clamp(covariance, -1.0, 1.0)
        ic_list.append(ic)
    
    if len(ic_list) == 0:
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    avg_ic = torch.mean(torch.stack(ic_list))
    loss = 1.0 - avg_ic
    
    return torch.clamp(loss, 0.0, 2.0)

class FastTransformerTrainer:
    """快速Transformer训练器"""
    
    def __init__(self, model_config, max_stocks=None):
        self.model_config = model_config
        self.max_stocks = max_stocks
        self.device = device
        
        # 初始化快速模型
        self.model = FastTransformerModel(**model_config['model_params']).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=model_config['training_params']['learning_rate'],
            weight_decay=model_config['training_params']['weight_decay'],
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=model_config['training_params']['num_epochs']
        )
        
        print(f"✓ 快速模型初始化完成")
        print(f"✓ 设备: {self.device}")
    
    def load_direct_data(self, data_file='direct_features_latest.csv'):
        """加载数据"""
        print("正在加载数据...")
        
        if not os.path.exists(data_file):
            print(f"数据文件不存在: {data_file}")
            return None
        
        try:
            features_df = pd.read_csv(data_file)
            features_df['end_date'] = pd.to_datetime(features_df['end_date'])
            features_df['StockID'] = features_df['StockID'].astype('category')
            print(f"✓ 特征数据加载成功: {features_df.shape}")
        except Exception as e:
            print(f"特征数据加载失败: {e}")
            return None
        
        try:
            stock_data = pd.read_feather('stock_price_vol_d.txt')
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data['StockID'] = stock_data['StockID'].astype('category')
            print(f"✓ 价格数据加载成功: {stock_data.shape}")
        except Exception as e:
            print(f"价格数据加载失败: {e}")
            return None
        
        return features_df, stock_data
    
    def process_features_and_labels(self, features_df, stock_data, sampling_interval=5):
        """快速数据处理"""
        print("正在构建训练数据...")
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        seq_len = 40
        label_horizon = 10
        
        # 预排序
        features_df = features_df.sort_values(['StockID', 'end_date']).reset_index(drop=True)
        stock_data = stock_data.sort_values(['StockID', 'date']).reset_index(drop=True)
        
        # 筛选有效股票
        stock_counts = features_df['StockID'].value_counts()
        valid_stocks = stock_counts[stock_counts >= 40].index.tolist()  # 降低要求
        
        if self.max_stocks:
            valid_stocks = valid_stocks[:self.max_stocks]
        
        print(f"✓ 有效股票数量: {len(valid_stocks)} 只")
        
        # 预分配内存 - 减少内存使用
        max_samples = min(2000000, sum(len(features_df[features_df['StockID'] == stock]) - seq_len - label_horizon + 1 
                         for stock in valid_stocks) // sampling_interval)
        
        features_array = np.zeros((max_samples, seq_len, len(feature_cols)), dtype=np.float32)
        labels_array = np.zeros((max_samples, label_horizon), dtype=np.float32)
        dates_array = np.zeros(max_samples, dtype=object)
        
        sample_idx = 0
        
        for stock_id in tqdm(valid_stocks, desc="处理股票"):
            stock_features = features_df[features_df['StockID'] == stock_id]
            stock_prices = stock_data[stock_data['StockID'] == stock_id]
            
            if len(stock_features) < seq_len + label_horizon or len(stock_prices) < seq_len + label_horizon:
                continue
            
            # 创建日期映射
            price_dates = stock_prices['date'].values
            price_close = stock_prices['close'].values
            date_to_idx = {date: idx for idx, date in enumerate(price_dates)}
            
            feature_values = stock_features[feature_cols].values
            feature_dates = stock_features['end_date'].values
            
            for i in range(0, len(stock_features) - seq_len - label_horizon + 1, sampling_interval):
                if sample_idx >= max_samples:
                    break
                
                # 特征序列
                feature_seq = feature_values[i:i+seq_len]
                
                # 标签序列
                feature_end_date = feature_dates[i+seq_len-1]
                if feature_end_date not in date_to_idx:
                    continue
                
                feature_end_idx = date_to_idx[feature_end_date]
                t0_idx = feature_end_idx + 1
                
                if t0_idx + label_horizon >= len(price_close):
                    continue
                
                # 计算收益率
                future_prices = price_close[t0_idx:t0_idx+label_horizon+1]
                returns = (future_prices[1:] - future_prices[:-1]) / future_prices[:-1]
                
                if len(returns) != label_horizon:
                    continue
                
                # 异常值处理
                if np.any(np.abs(returns) > 0.5):
                    continue
                
                features_array[sample_idx] = feature_seq
                labels_array[sample_idx] = returns
                dates_array[sample_idx] = feature_end_date
                sample_idx += 1
        
        if sample_idx == 0:
            print("没有找到有效数据")
            return None, None, None
        
        features = features_array[:sample_idx]
        labels = labels_array[:sample_idx]
        dates = dates_array[:sample_idx]
        
        print(f"✓ 最终数据形状: {features.shape}")
        return features, labels, dates
    
    def create_data_loaders(self, data, batch_size=2048):
        """创建优化的数据加载器"""
        features, labels, dates = data
        
        # 按时间排序
        sorted_indices = np.argsort(dates)
        features = features[sorted_indices]
        labels = labels[sorted_indices]
        dates = dates[sorted_indices]
        
        # 时间分割
        split_idx = int(0.8 * len(features))
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features[split_idx:]
        val_labels = labels[split_idx:]
        
        # 快速标准化
        train_mean = np.mean(train_features, axis=(0, 1), keepdims=True)
        train_std = np.std(train_features, axis=(0, 1), keepdims=True) + 1e-8
        
        train_features_scaled = (train_features - train_mean) / train_std
        val_features_scaled = (val_features - train_mean) / train_std
        
        # 创建数据集
        train_dataset = FactorDataset(train_features_scaled, train_labels)
        val_dataset = FactorDataset(val_features_scaled, val_labels)
        
        # 优化的数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=8,  # 增加worker数量
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """快速训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc="训练"):
            batch_features = batch_features.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            predictions = self.model(batch_features, batch_labels, teacher_forcing=True)
            loss = fast_ic_loss(predictions, batch_labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader):
        """快速验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_loader, desc="验证"):
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                predictions = self.model(batch_features, batch_labels, teacher_forcing=False)
                loss = fast_ic_loss(predictions, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # 快速IC计算
        if all_predictions and all_labels:
            all_preds = np.concatenate(all_predictions, axis=0)
            all_labs = np.concatenate(all_labels, axis=0)
            
            ic_list = []
            for t in range(min(3, all_preds.shape[1])):  # 只计算前3个时间步
                pred = all_preds[:, t]
                true = all_labs[:, t]
                
                pred = np.where(np.isnan(pred) | np.isinf(pred), 0.0, pred)
                true = np.where(np.isnan(true) | np.isinf(true), 0.0, true)
                
                pred_mean = np.mean(pred)
                pred_std = np.std(pred)
                true_mean = np.mean(true)
                true_std = np.std(true)
                
                if pred_std > 1e-8 and true_std > 1e-8:
                    pred_norm = (pred - pred_mean) / pred_std
                    true_norm = (true - true_mean) / true_std
                    ic = np.mean(pred_norm * true_norm)
                    ic_list.append(ic)
            
            avg_ic = np.mean(ic_list) if ic_list else 0.0
        else:
            avg_ic = 0.0
        
        return avg_loss, avg_ic
    
    def train(self, train_loader, val_loader, num_epochs=200):
        """快速训练"""
        print("开始快速训练...")
        
        best_val_ic = -float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 30
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_ic = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}/{val_loss:.4f}, IC={val_ic:.4f}")
            
            # 保存最佳模型
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"✓ 发现更好的IC: {best_val_ic:.6f}")
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_ic': best_val_ic,
                    'config': self.model_config
                }, 'fast_best_model.pth')
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"早停: 最佳IC={best_val_ic:.6f}")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return best_val_ic
    
    def train_ensemble_models(self, seeds=[123, 456, 789]):
        """训练快速集成模型"""
        print("训练快速集成模型...")
        
        os.makedirs('trained_models', exist_ok=True)
        
        models_info = []
        
        # 加载数据
        data = self.load_direct_data()
        if data is None:
            return None
        
        features_df, stock_data = data
        features, labels, dates = self.process_features_and_labels(features_df, stock_data)
        if features is None:
            return None
        
        train_loader, val_loader = self.create_data_loaders(
            (features, labels, dates), 
            batch_size=self.model_config['training_params']['batch_size']
        )
        
        for i, seed in enumerate(seeds):
            print(f"\n训练第 {i+1}/{len(seeds)} 个模型 (种子: {seed})")
            
            # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # 重新初始化模型
            self.model = FastTransformerModel(**self.model_config['model_params']).to(self.device)
            
            # 重新初始化优化器和调度器
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.model_config['training_params']['learning_rate'],
                weight_decay=self.model_config['training_params']['weight_decay'],
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.model_config['training_params']['num_epochs']
            )
            
            # 训练
            best_ic = self.train(train_loader, val_loader, num_epochs=200)
            
            # 保存模型
            model_file = f"trained_models/fast_model_seed_{seed}.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'best_ic': best_ic,
                'config': self.model_config
            }, model_file)
            
            models_info.append({
                'model_id': i + 1,
                'seed': seed,
                'best_ic': float(best_ic),
                'model_file': model_file
            })
            
            print(f"✓ 模型 {i+1} 完成，最佳IC: {best_ic:.6f}")
        
        # 保存集成信息
        ensemble_info = {
            'models': models_info,
            'ensemble_type': 'equal_weight',
            'num_models': len(seeds),
            'training_date': datetime.now().isoformat()
        }
        
        with open('trained_models/fast_ensemble_info.json', 'w') as f:
            json.dump(ensemble_info, f, indent=4)
        
        avg_ic = np.mean([info['best_ic'] for info in models_info])
        print(f"\n✓ 快速集成训练完成！平均IC: {avg_ic:.6f}")
        
        return models_info

def main():
    """主函数"""
    # 快速模型配置
    model_config = {
        'model_params': {
            'input_dim': 6,
            'd_model': 256,  # 适中的模型容量
            'nhead': 8,
            'num_layers': 4,  # 适中的层数
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'label_horizon': 10
        },
        'training_params': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'batch_size': 2048,  # 增大批次大小
            'num_epochs': 200
        }
    }
    
    print("快速Transformer因子训练")
    print("="*60)
    print("速度优化:")
    print("✓ 适中的模型容量 (d_model=256, layers=4)")
    print("✓ 简化损失函数")
    print("✓ 增大批次大小 (2048)")
    print("✓ 优化数据加载")
    print("✓ 使用ReLU激活函数")
    print("✓ 减少数据量 (200万样本)")
    print("="*60)
    
    # 初始化训练器
    trainer = FastTransformerTrainer(model_config)
    
    # 训练集成模型
    ensemble_result = trainer.train_ensemble_models(seeds=[123, 456, 789])
    
    if ensemble_result:
        print(f"\n✓ 快速训练完成！")
        print(f"✓ 平均IC: {np.mean([info['best_ic'] for info in ensemble_result]):.6f}")

if __name__ == "__main__":
    main() 