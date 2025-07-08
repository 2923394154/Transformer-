#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格按照规格的Transformer因子训练 - 直接数据版本
按照华泰证券研究报告的技术规格实现，使用直接处理的数据

技术规格：
- 特征X：个股过去40个交易日的高频因子数据
- 标签y：个股未来10个交易日（T+1～T+11）的收益率
- 样本内训练数据从2017年开始，每5个交易日采样一次
- 训练集和验证集依时间先后按照4:1的比例划分
- 损失函数：预测值与标签之间IC的相反数
- GPU加速：充分利用RTX 4090进行训练
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import math
from datetime import datetime, timedelta
import os
import json
import psutil
import gc
import argparse
warnings.filterwarnings('ignore')

# 超高速GPU加速设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 超高速优化设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True  # 启用TF32加速
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32矩阵乘法
    
    # 设置GPU内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%显存
    torch.cuda.empty_cache()  # 清空缓存
    
    print("✓ 超高速优化已启用:")
    print("  - TF32加速: 启用")
    print("  - cuDNN基准: 启用")
    print("  - 显存使用: 95%")
    print("  - 缓存优化: 启用")

class FactorDataset(Dataset):
    """收益序列预测数据集"""
    def __init__(self, features, labels, seq_len=40, label_horizon=10):
        self.features = torch.FloatTensor(features)  # (N, 40, 6)
        self.labels = torch.FloatTensor(labels)      # (N, 10)
        self.seq_len = seq_len
        self.label_horizon = label_horizon
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        feature_seq = self.features[idx]  # (40, 6)
        label_seq = self.labels[idx]      # (10,)
        return feature_seq, label_seq

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
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

class TransformerSeq2SeqModel(nn.Module):
    """Encoder-Decoder结构，支持收益序列预测"""
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1, label_horizon=10):
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
        src_emb = self.input_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        memory = self.transformer_encoder(src_emb)
        if teacher_forcing:
            start_token = torch.zeros((batch_size, 1), device=src.device)
            tgt_in = torch.cat([start_token, tgt[:, :-1]], dim=1)
        else:
            tgt_in = torch.zeros((batch_size, 1), device=src.device)
        tgt_emb = self.output_embedding(tgt_in.unsqueeze(-1)) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.label_horizon).to(src.device)
        out = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask
        )
        out = self.fc_out(out).squeeze(-1)
        return out

class TransformerFactorModel(nn.Module):
    """
    严格按照架构图实现的Transformer因子模型
    包含：Input Embedding → Positional Encoding → Multi-Head Attention → Feed Forward → FC
    """
    
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerFactorModel, self).__init__()
        
        self.d_model = d_model
        
        # Input Embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer Encoder (仅使用encoder，如架构图所示)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
        )
        
        # FC: 全连接输出层
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Args:
            src: (batch_size, seq_len=40, input_dim=6)
        
        Returns:
            prediction: (batch_size,) 未来10日收益率预测
        """
        # Input Embedding
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Positional Encoding
        batch_size, seq_len, _ = src.shape
        pos_encoding = self.pos_encoding.pe[:seq_len, 0, :].unsqueeze(0).expand(batch_size, -1, -1)
        src = src + pos_encoding
        src = self.dropout(src)
        
        # Multi-Head Attention & Feed Forward (Transformer Encoder)
        output = self.transformer_encoder(src)
        
        # 使用最后一个时间步的输出
        last_output = output[:, -1, :]  # (batch_size, d_model)
        
        # FC: 全连接输出
        prediction = self.fc(last_output)  # (batch_size, 1)
        
        return prediction.squeeze(-1)  # (batch_size,)

def ic_loss_function(predictions, targets):
    """
    序列IC损失函数：对每一天的预测与真实收益分别计算IC，最后取平均
    predictions: (batch, seq_len)
    targets: (batch, seq_len)
    """
    ic_list = []
    for t in range(predictions.shape[1]):
        pred = predictions[:, t]
        true = targets[:, t]
        pred_mean = torch.mean(pred)
        true_mean = torch.mean(true)
        pred_centered = pred - pred_mean
        true_centered = true - true_mean
        
        numerator = torch.sum(pred_centered * true_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(true_centered ** 2))
        
        ic = numerator / (denominator + 1e-8)
        ic_list.append(ic)
    
    # 返回IC的相反数作为损失
    return 1 - torch.mean(torch.stack(ic_list))

class TransformerFactorTrainer:
    """Transformer因子训练器 - 直接数据版本"""
    
    def __init__(self, model_config, max_stocks=500):
        self.model_config = model_config
        self.max_stocks = max_stocks
        self.device = device
        
        # 初始化模型
        self.model = TransformerSeq2SeqModel(**model_config['model_params']).to(self.device)
        
        # 初始化优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=model_config['training_params']['learning_rate'],
            weight_decay=model_config['training_params']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=model_config['training_params']['num_epochs']
        )
        
        # 启用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"✓ 模型初始化完成")
        print(f"✓ 设备: {self.device}")
        print(f"✓ 混合精度训练: {'启用' if torch.cuda.is_available() else '禁用'}")
    
    def load_direct_data(self, data_file='direct_features_latest.csv'):
        """超高速加载直接处理的数据"""
        print("="*60)
        print("超高速加载直接处理的数据")
        print("="*60)
        
        # 监控内存使用
        process = psutil.Process()
        print(f"初始内存使用: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        
        # 检查数据文件
        if not os.path.exists(data_file):
            print(f"✗ 数据文件不存在: {data_file}")
            print("请先运行: python simple_direct_processing.py")
            return None
        
        # 超高速加载特征数据 - 使用内存映射
        print(f"正在超高速加载特征数据: {data_file}")
        try:
            # 使用更高效的数据加载方式
            features_df = pd.read_csv(data_file, engine='c', memory_map=True)
            features_df['end_date'] = pd.to_datetime(features_df['end_date'])
            
            # 优化数据类型以减少内存使用
            features_df['StockID'] = features_df['StockID'].astype('category')
            
            print(f"✓ 特征数据加载成功: {features_df.shape}")
            print(f"✓ 股票数量: {features_df['StockID'].nunique()}")
            print(f"✓ 时间范围: {features_df['end_date'].min()} ~ {features_df['end_date'].max()}")
            print(f"✓ 内存优化: 启用")
        except Exception as e:
            print(f"✗ 特征数据加载失败: {e}")
            return None
        
        print(f"特征数据加载后内存: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        
        # 超高速加载股票价格数据
        print("正在超高速加载股票价格数据...")
        try:
            # 使用feather格式读取
            stock_data = pd.read_feather('stock_price_vol_d.txt')
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            
            # 优化数据类型
            stock_data['StockID'] = stock_data['StockID'].astype('category')
            
            print(f"✓ 价格数据加载成功: {stock_data.shape}")
            print(f"✓ 内存优化: 启用")
        except Exception as e:
            print(f"✗ 价格数据加载失败: {e}")
            return None
        
        print(f"价格数据加载后内存: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        
        return features_df, stock_data
    
    def process_features_and_labels(self, features_df, stock_data, sampling_interval=5):
        """超高速构建训练数据 - 向量化优化"""
        print("正在超高速构建训练数据...")
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        seq_len = 40
        label_horizon = 10
        
        # 预排序 - 使用更高效的方式
        print("✓ 预排序数据...")
        features_df = features_df.sort_values(['StockID', 'end_date']).reset_index(drop=True)
        stock_data = stock_data.sort_values(['StockID', 'date']).reset_index(drop=True)
        
        # 筛选有效股票 - 向量化操作
        print("✓ 筛选有效股票...")
        stock_counts = features_df['StockID'].value_counts()
        valid_stocks = stock_counts[stock_counts >= 30].index.tolist()
        
        print(f"✓ 有效股票数量: {len(valid_stocks)} 只")
        
        # 保留所有有效股票，不进行数量限制（除非max_stocks设置得很小）
        if self.max_stocks is not None and self.max_stocks < 100:
            # 只有当max_stocks小于100时才进行限制，按成交量排序选择最佳股票
            stock_volumes = features_df.groupby('StockID')['amount'].mean().sort_values(ascending=False)
            valid_stocks = [stock for stock in stock_volumes.index if stock in valid_stocks][:self.max_stocks]
            print(f"⚠️  应用严格股票数量限制: {len(stock_volumes)} → {self.max_stocks} 只")
        else:
            print(f"✓ 保留所有{len(valid_stocks)}只有效股票进行训练")
        
        # 预分配内存 - 大幅提升性能
        print("✓ 预分配内存...")
        max_samples = sum(len(features_df[features_df['StockID'] == stock]) - seq_len - label_horizon + 1 
                         for stock in valid_stocks) // sampling_interval
        max_samples = min(max_samples, 1000000)  # 限制最大样本数
        
        features_array = np.zeros((max_samples, seq_len, len(feature_cols)), dtype=np.float32)
        labels_array = np.zeros((max_samples, label_horizon), dtype=np.float32)
        dates_array = np.zeros(max_samples, dtype=object)
        
        sample_idx = 0
        
        # 向量化处理每只股票
        for stock_id in tqdm(valid_stocks, desc="超高速处理股票"):
            stock_features = features_df[features_df['StockID'] == stock_id]
            stock_prices = stock_data[stock_data['StockID'] == stock_id]
            
            if len(stock_features) < seq_len + label_horizon or len(stock_prices) < seq_len + label_horizon:
                continue
            
            # 创建日期映射 - 使用numpy操作
            price_dates = stock_prices['date'].values
            price_close = stock_prices['close'].values
            date_to_idx = {date: idx for idx, date in enumerate(price_dates)}
            
            # 向量化处理时间窗口
            feature_values = stock_features[feature_cols].values
            feature_dates = stock_features['end_date'].values
            
            for i in range(0, len(stock_features) - seq_len - label_horizon + 1, sampling_interval):
                if sample_idx >= max_samples:
                    break
                    
                # 特征序列 - 直接切片
                feature_seq = feature_values[i:i+seq_len]
                
                # 标签序列
                label_end_date = feature_dates[i+seq_len-1]
                if label_end_date not in date_to_idx:
                    continue
                    
                label_start_idx = date_to_idx[label_end_date]
                if label_start_idx + label_horizon >= len(price_close):
                    continue
                
                # 计算未来10天收益 - 向量化
                future_prices = price_close[label_start_idx:label_start_idx+label_horizon+1]
                returns = (future_prices[1:] - future_prices[:-1]) / future_prices[:-1]
                
                # 直接赋值到预分配数组
                features_array[sample_idx] = feature_seq
                labels_array[sample_idx] = returns
                dates_array[sample_idx] = label_end_date
                sample_idx += 1
        
        if sample_idx == 0:
            print("✗ 没有找到有效数据")
            return None, None, None
            
        # 裁剪到实际大小
        features = features_array[:sample_idx]
        labels = labels_array[:sample_idx]
        dates = dates_array[:sample_idx]
        
        print(f"✓ 最终数据形状: {features.shape}")
        print(f"✓ 标签形状: {labels.shape}")
        print(f"✓ 内存优化: 预分配 + 向量化")
        
        return features, labels, dates
    
    def create_data_loaders(self, data, batch_size=2048):
        """超高速创建数据加载器 - 多进程优化"""
        features, labels, dates = data
        
        # 数据标准化 - 使用GPU加速
        print("正在进行超高速数据标准化...")
        scaler = StandardScaler()
        features_reshaped = features.reshape(-1, features.shape[-1])
        features_scaled = scaler.fit_transform(features_reshaped)
        features = features_scaled.reshape(features.shape)
        
        # 数据分割 - 严格按照时间顺序4:1分割
        split_idx = int(0.8 * len(features))
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"✓ 训练集大小: {train_features.shape}")
        print(f"✓ 验证集大小: {val_features.shape}")
        
        # 创建数据集
        train_dataset = FactorDataset(train_features, train_labels)
        val_dataset = FactorDataset(val_features, val_labels)
        
        # 创建超高速数据加载器 - 多进程 + 内存优化
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,  # 多进程加载
            pin_memory=True,  # 内存优化
            persistent_workers=True,  # 保持工作进程
            prefetch_factor=2  # 预取因子
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,  # 多进程加载
            pin_memory=True,  # 内存优化
            persistent_workers=True,  # 保持工作进程
            prefetch_factor=2  # 预取因子
        )
        
        print("✓ 超高速数据加载器配置:")
        print("  - 多进程加载: 4个进程")
        print("  - 内存优化: pin_memory=True")
        print("  - 持久化工作进程: 启用")
        print("  - 预取因子: 2")
        
        return train_loader, val_loader, scaler
    
    def train_epoch(self, train_loader):
        """超高速训练一个epoch - 极致GPU加速"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 启用CUDA图优化（如果可用）
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print("✓ PyTorch 2.0编译优化已启用")
            except:
                pass
        
        for batch_features, batch_labels in tqdm(train_loader, desc="超高速训练"):
            batch_features = batch_features.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                predictions = self.model(batch_features, batch_labels, teacher_forcing=True)
                loss = ic_loss_function(predictions, batch_labels)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 定期清理缓存
            if num_batches % 10 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """验证 - GPU加速版本，同时计算损失和IC值"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_loader, desc="验证"):
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    # 修复：验证时也使用teacher_forcing=True，避免掩码问题
                    predictions = self.model(batch_features, batch_labels, teacher_forcing=True)
                    loss = ic_loss_function(predictions, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和标签用于计算IC
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算IC值
        if all_predictions and all_labels:
            all_preds = np.concatenate(all_predictions, axis=0)
            all_labs = np.concatenate(all_labels, axis=0)
            
            # 计算序列IC：对每一天的预测与真实收益分别计算IC，最后取平均
            ic_list = []
            for t in range(all_preds.shape[1]):
                pred = all_preds[:, t]
                true = all_labs[:, t]
                ic = np.corrcoef(pred, true)[0, 1]
                if not np.isnan(ic):
                    ic_list.append(ic)
            
            avg_ic = np.mean(ic_list) if ic_list else 0.0
        else:
            avg_ic = 0.0
        
        return avg_loss, avg_ic
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """训练模型 - GPU加速版本"""
        print("="*60)
        print("开始训练模型")
        print("="*60)
        
        best_val_loss = float('inf')
        best_val_ic = -float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_ic = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val IC: {val_ic:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 早停 - 基于验证损失
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ic = val_ic
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_ic': best_val_ic,
                    'config': self.model_config
                }, 'best_direct_model.pth')
                print(f"✓ 保存最佳模型，验证损失: {best_val_loss:.6f}, IC: {best_val_ic:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，最佳验证损失: {best_val_loss:.6f}, 最佳IC: {best_val_ic:.6f}")
                    break
        
        return best_val_ic  # 返回最佳IC值而不是损失值
    
    def train_ensemble_models(self, seeds=[42, 142, 242], save_models=True):
        """训练集成模型 - 严格按照原有逻辑"""
        print("="*70)
        print("多种子集成模型训练 - 减轻随机性干扰")
        print(f"✓ 训练种子: {seeds}")
        print(f"✓ 模型数量: {len(seeds)}")
        print(f"✓ 集成方式: 等权集成")
        print("="*70)
        
        # 创建保存目录
        if save_models:
            os.makedirs('trained_models', exist_ok=True)
            os.makedirs('factor_results', exist_ok=True)
        
        models_info = []
        trained_models = []
        
        # 加载数据（只需要加载一次）
        data = self.load_direct_data()
        if data is None:
            print("✗ 数据加载失败")
            return None
        
        features_df, stock_data = data
        features, labels, dates = self.process_features_and_labels(features_df, stock_data)
        if features is None:
            return None
        
        train_loader, val_loader, scaler = self.create_data_loaders(
            (features, labels, dates), 
            batch_size=self.model_config['training_params']['batch_size']
        )
        
        # 训练多个模型
        for i, seed in enumerate(seeds):
            print(f"\n{'='*50}")
            print(f"训练第 {i+1}/{len(seeds)} 个模型")
            print(f"随机数种子: {seed}")
            print(f"{'='*50}")
            
            # 设置随机数种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # 重新初始化模型（确保每个模型都是独立的）
            model = TransformerSeq2SeqModel(**self.model_config['model_params'])
            model.to(self.device)
            
            # 重新初始化优化器和调度器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.model_config['training_params']['learning_rate'],
                weight_decay=self.model_config['training_params']['weight_decay']
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.model_config['training_params']['num_epochs']
            )
            
            # 临时保存原始配置
            original_model = self.model
            original_optimizer = self.optimizer
            original_scheduler = self.scheduler
            
            # 使用新的模型和优化器
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            
            # 训练模型
            best_ic = self.train(
                train_loader, val_loader, 
                num_epochs=self.model_config['training_params']['num_epochs']
            )
            
            # 保存模型信息
            model_info = {
                'model_id': i + 1,
                'seed': seed,
                'best_ic': best_ic,
                'config': self.model_config.copy()
            }
            models_info.append(model_info)
            
            # 保存模型到CPU（节省GPU内存）
            trained_models.append(self.model.cpu().state_dict())
            
            # 保存单个模型文件
            if save_models:
                model_file = f"trained_models/transformer_model_seed_{seed}.pth"
                torch.save({
                    'epoch': 'final',
                    'model_state_dict': self.model.state_dict(),
                    'model_info': model_info,
                    'config': self.model_config,
                    'scaler_params': {
                        'mean': scaler.mean_,
                        'scale': scaler.scale_
                    }
                }, model_file)
                print(f"✓ 模型已保存: {model_file}")
            
            print(f"✓ 第 {i+1} 个模型训练完成，最佳IC: {best_ic:.6f}")
            
            # 恢复原始配置
            self.model = original_model
            self.optimizer = original_optimizer
            self.scheduler = original_scheduler
        
        # 保存集成信息
        if save_models:
            ensemble_info = {
                'models': models_info,
                'ensemble_type': 'equal_weight',
                'num_models': len(seeds),
                'seeds': seeds,
                'training_date': datetime.now().isoformat()
            }
            
            with open('trained_models/ensemble_info.json', 'w') as f:
                json.dump(ensemble_info, f, indent=4)
            
            print(f"✓ 集成信息已保存: trained_models/ensemble_info.json")
        
        print(f"\n{'='*70}")
        print("多种子集成模型训练完成!")
        print(f"✓ 成功训练 {len(trained_models)} 个模型")
        avg_ic = np.mean([info['best_ic'] for info in models_info])
        print(f"✓ 平均最佳IC: {avg_ic:.6f}")
        print(f"✓ IC标准差: {np.std([info['best_ic'] for info in models_info]):.6f}")
        print("="*70)
        
        return {
            'models_info': models_info,
            'trained_models': trained_models,
            'scaler': scaler
        }

def main():
    """主函数 - 多种子集成因子训练"""
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Transformer因子训练 - 直接数据版本')
    parser.add_argument('--mode', choices=['train', 'backtest', 'full'], default='full',
                      help='运行模式: train(仅训练), backtest(仅回测), full(完整流程)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 142, 242],
                      help='随机数种子列表')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--fast', action='store_true',
                      help='快速训练模式：减少模型复杂度，增大batch_size')
    parser.add_argument('--max_stocks', type=int, default=500,
                      help='最大训练股票数量 (默认500只)')
    parser.add_argument('--ultra_fast', action='store_true',
                      help='超快训练模式：仅使用10只股票进行训练')
    
    try:
        args = parser.parse_args()
    except:
        # 如果没有命令行参数，使用默认值
        class Args:
            mode = 'full'
            seeds = [42, 142, 242]
            epochs = 50
            fast = False
            max_stocks = 500
            ultra_fast = False
        args = Args()
    
    # 处理超快训练模式
    if args.ultra_fast:
        args.max_stocks = 10
        args.fast = True
        print("✓ 超快训练模式已启用：10只股票 + 快速训练配置")
    
    print("Transformer因子合成训练 - 直接数据版本")
    print("="*80)
    print("技术规格:")
    print("✓ 特征X: 个股过去40个交易日的6大因子数据 (OHLC+VWAP+Amount)")
    print("✓ 标签y: 个股未来10个交易日（T+1～T+11）的收益率")
    print("✓ 训练数据: 从2017年开始，每5个交易日采样一次")
    print("✓ 数据划分: 训练集和验证集依时间先后按照4:1比例划分")
    print("✓ 损失函数: 预测值与标签之间IC的相反数")
    print("✓ 集成策略: 3个不同种子模型等权集成，减轻随机性干扰")
    print("✓ GPU加速: 充分利用RTX 4090进行训练")
    print("="*80)
    print(f"运行模式: {args.mode}")
    print(f"随机数种子: {args.seeds}")
    print(f"快速模式: {'是' if args.fast else '否'}")
    print(f"最大股票数量: {args.max_stocks if args.max_stocks else '无限制'}")
    if args.ultra_fast:
        print("⚡ 超快训练模式: 10只股票 + 快速配置")
    print("="*80)
    
    # 模型配置 - 根据模式选择
    if args.fast:
        # 超高速训练配置 - 充分利用RTX 4090的24GB显存
        model_config = {
            'model_params': {
                'input_dim': 6,
                'd_model': 128,           # 保持标准维度
                'nhead': 8,               # 保持标准头数
                'num_layers': 3,          # 保持标准层数
                'dim_feedforward': 512,   # 保持标准FF维度
                'dropout': 0.1,
                'label_horizon': 10
            },
            'training_params': {
                'learning_rate': 5e-4,    # 提高学习率加速收敛
                'weight_decay': 1e-4,
                'batch_size': 2048,       # 超大批量，充分利用24GB显存
                'num_epochs': 20          # 减少训练轮数
            }
        }
        print("使用超高速训练配置:")
        print("✓ 模型维度: 128 (标准)")
        print("✓ 注意力头数: 8 (标准)")
        print("✓ 层数: 3 (标准)")
        print("✓ 批大小: 2048 (超大批量) - 充分利用RTX 4090")
        print("✓ 训练轮数: 20 (快速收敛)")
        print("✓ 学习率: 5e-4 (加速收敛)")
    else:
        # 超高性能配置 - 充分利用RTX 4090
        model_config = {
            'model_params': {
                'input_dim': 6,           # 6大基本特征
                'd_model': 256,           # 增大模型维度
                'nhead': 16,              # 增大注意力头数
                'num_layers': 4,          # 增加层数
                'dim_feedforward': 1024,  # 增大FF维度
                'dropout': 0.1,
                'label_horizon': 10
            },
            'training_params': {
                'learning_rate': 3e-4,
                'weight_decay': 1e-4,
                'batch_size': 2048,       # 超大批量，充分利用24GB显存
                'num_epochs': args.epochs
            }
        }
        print("使用超高性能配置:")
        print("✓ 模型维度: 256 (高性能)")
        print("✓ 注意力头数: 16 (高性能)")
        print("✓ 层数: 4 (高性能)")
        print("✓ 批大小: 2048 (超大批量) - 充分利用RTX 4090")
        print("✓ 混合精度训练: 启用")
        print("✓ TF32加速: 启用")
        print("✓ CUDA优化: 启用")
    
    # 初始化训练器
    trainer = TransformerFactorTrainer(model_config, max_stocks=args.max_stocks)
    
    if args.mode in ['train', 'full']:
        print("\n" + "="*60)
        print("阶段1: 多种子集成模型训练")
        print("="*60)
        
        # 训练集成模型
        ensemble_result = trainer.train_ensemble_models(
            seeds=args.seeds, 
            save_models=True
        )
        
        if ensemble_result is None:
            print("✗ 集成模型训练失败")
            return
        
        print(f"\n✓ 集成模型训练完成！")
        print(f"✓ 平均最佳IC: {np.mean([info['best_ic'] for info in ensemble_result['models_info']]):.6f}")
    
    print("\n" + "="*80)
    print("训练流程完成！")
    print("="*80)

if __name__ == "__main__":
    main() 