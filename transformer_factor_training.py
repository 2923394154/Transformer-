#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格按照规格的Transformer因子训练
按照华泰证券研究报告的技术规格实现

技术规格：
- 特征X：个股过去40个交易日的高频因子数据
- 标签y：个股未来10个交易日（T+1～T+11）的收益率
- 样本内训练数据从2013年开始，每5个交易日采样一次
- 训练集和验证集依时间先后按照4:1的比例划分
- 损失函数：预测值与标签之间IC的相反数
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
warnings.filterwarnings('ignore')

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
    
    def __init__(self, d_model, max_len=5000):
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
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
        true_std = torch.sqrt(torch.sum(true_centered ** 2))
        denominator = pred_std * true_std + 1e-8
        ic = numerator / denominator
        ic_list.append(ic)
    ic_tensor = torch.stack(ic_list)
    return -ic_tensor.mean()

class TransformerFactorTrainer:
    """严格按照规格的Transformer因子训练器"""
    
    def __init__(self, model_config):
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 启用混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 初始化模型
        self.model = TransformerSeq2SeqModel(**model_config['model_params']).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=model_config['training_params']['learning_rate'],
            weight_decay=model_config['training_params']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=model_config['training_params']['num_epochs']
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_ics = []
        
        # 设置CUDA优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("✓ 启用CUDA优化: cudnn.benchmark=True")
        
    def load_and_process_data(self, start_date="2013-01-01", sampling_interval=5):
        """
        按照规格加载和处理数据
        
        Args:
            start_date: 训练数据开始日期 (2013年开始)
            sampling_interval: 采样间隔 (每5个交易日采样一次)
        """
        print("="*60)
        print("按照规格加载和处理因子数据")
        print(f"✓ 开始日期: {start_date}")
        print(f"✓ 采样间隔: 每{sampling_interval}个交易日")
        print(f"✓ 特征窗口: 过去40个交易日")
        print(f"✓ 标签窗口: 未来10个交易日")
        print("="*60)
        
        # 加载等量K线特征数据
        try:
            features_df = pd.read_csv('equal_volume_features_all.csv')
            print(f"✓ 加载特征数据: {features_df.shape}")
        except:
            print("✗ 无法加载特征数据，请先运行等量K线生成")
            return None
        
        # 加载股票价格数据
        try:
            stock_data = pd.read_feather('stock_price_vol_d.txt')
            print(f"✓ 加载价格数据: {stock_data.shape}")
        except:
            print("✗ 无法加载价格数据")
            return None
        
        # 时间过滤：从2013年开始
        features_df['end_date'] = pd.to_datetime(features_df['end_date'])
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        start_date = pd.to_datetime(start_date)
        features_df = features_df[features_df['end_date'] >= start_date]
        stock_data = stock_data[stock_data['date'] >= start_date]
        
        print(f"✓ 时间过滤后 - 特征数据: {features_df.shape}, 价格数据: {stock_data.shape}")
        
        # 处理特征和标签 - 使用快速版本
        features, labels, dates = self._process_features_and_labels_fast(
            features_df, stock_data, sampling_interval
        )
        
        if features is None:
            return None
        
        # 按时间顺序划分训练集和验证集 (4:1)
        split_idx = int(len(features) * 0.8)
        
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        val_features = features[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"✓ 训练集: {train_features.shape}, 验证集: {val_features.shape}")
        print(f"✓ 训练标签: {train_labels.shape}, 验证标签: {val_labels.shape}")
        
        # 特征标准化 - 处理3维数组
        scaler = StandardScaler()
        # 重塑为2维进行标准化
        train_features_reshaped = train_features.reshape(-1, train_features.shape[-1])
        val_features_reshaped = val_features.reshape(-1, val_features.shape[-1])
        
        train_features_scaled = scaler.fit_transform(train_features_reshaped)
        val_features_scaled = scaler.transform(val_features_reshaped)
        
        # 重塑回3维
        train_features_scaled = train_features_scaled.reshape(train_features.shape)
        val_features_scaled = val_features_scaled.reshape(val_features.shape)
        
        print(f"✓ 特征标准化完成")
        
        return {
            'train_features': train_features_scaled,
            'train_labels': train_labels,
            'val_features': val_features_scaled,
            'val_labels': val_labels,
            'scaler': scaler
        }
    
    def _process_features_and_labels(self, features_df, stock_data, sampling_interval):
        """
        构建(X_seq, y_seq)对，y_seq为未来10天逐日收益序列 - 优化版本
        """
        print("正在构建收益序列预测数据...")
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        seq_len = 40
        label_horizon = 10
        
        # 预排序和索引优化
        print("✓ 预处理数据...")
        features_df = features_df.sort_values(['StockID', 'end_date']).reset_index(drop=True)
        stock_data = stock_data.sort_values(['StockID', 'date']).reset_index(drop=True)
        
        # 创建日期索引映射，避免重复查询
        print("✓ 创建日期索引映射...")
        date_mapping = {}
        for stock_id in stock_data['StockID'].unique():
            stock_prices = stock_data[stock_data['StockID'] == stock_id]
            date_mapping[stock_id] = dict(zip(stock_prices['date'], stock_prices.index))
        
        # 筛选有效股票
        valid_stocks = features_df['StockID'].value_counts()
        valid_stocks = valid_stocks[valid_stocks >= 100].index.tolist()[:50]  # 减少到50只股票
        print(f"处理 {len(valid_stocks)} 只股票...")
        
        features_list = []
        labels_list = []
        dates_list = []
        
        # 批量处理股票
        for stock_id in tqdm(valid_stocks, desc="处理股票"):
            # 获取该股票的所有特征和价格数据
            stock_features = features_df[features_df['StockID'] == stock_id]
            stock_prices = stock_data[stock_data['StockID'] == stock_id]
            
            if len(stock_features) < seq_len + label_horizon or len(stock_prices) < seq_len + label_horizon:
                continue
            
            # 创建特征日期到价格索引的映射
            feature_dates = stock_features['end_date'].values
            price_dates = stock_prices['date'].values
            price_close = stock_prices['close'].values
            
            # 使用numpy操作进行日期匹配
            date_to_idx = {date: idx for idx, date in enumerate(price_dates)}
            
            # 批量处理每个时间窗口
            for i in range(0, len(stock_features) - seq_len - label_horizon + 1, sampling_interval):
                # 特征序列
                feature_seq = stock_features.iloc[i:i+seq_len][feature_cols].values
                
                # 标签序列 - 使用预计算的索引
                label_end_date = stock_features.iloc[i+seq_len-1]['end_date']
                if label_end_date not in date_to_idx:
                    continue
                    
                label_start_idx = date_to_idx[label_end_date]
                if label_start_idx + label_horizon >= len(price_close):
                    continue
                
                # 计算逐日收益序列
                future_prices = price_close[label_start_idx:label_start_idx+label_horizon+1]
                returns = (future_prices[1:] - future_prices[:-1]) / future_prices[:-1]
                
                features_list.append(feature_seq)
                labels_list.append(returns)
                dates_list.append(label_end_date)
        
        if not features_list:
            print("✗ 没有找到有效数据")
            return None, None, None
            
        # 批量转换为numpy数组
        print("✓ 转换为numpy数组...")
        features = np.stack(features_list)
        labels = np.stack(labels_list)
        
        print(f"✓ 最终数据形状 - 特征: {features.shape}, 标签: {labels.shape}")
        return features, labels, dates_list
    
    def _process_features_and_labels_fast(self, features_df, stock_data, sampling_interval):
        """
        快速构建(X_seq, y_seq)对 - 使用向量化操作优化
        """
        print("正在快速构建收益序列预测数据...")
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        seq_len = 40
        label_horizon = 10
        
        # 预排序和索引优化
        print("✓ 预处理数据...")
        features_df = features_df.sort_values(['StockID', 'end_date']).reset_index(drop=True)
        stock_data = stock_data.sort_values(['StockID', 'date']).reset_index(drop=True)
        
        # 筛选有效股票
        valid_stocks = features_df['StockID'].value_counts()
        valid_stocks = valid_stocks[valid_stocks >= 100].index.tolist()[:50]  # 减少到50只股票
        print(f"处理 {len(valid_stocks)} 只股票...")
        
        features_list = []
        labels_list = []
        dates_list = []
        
        # 批量处理股票
        for stock_id in tqdm(valid_stocks, desc="处理股票"):
            # 获取该股票的所有特征和价格数据
            stock_features = features_df[features_df['StockID'] == stock_id]
            stock_prices = stock_data[stock_data['StockID'] == stock_id]
            
            if len(stock_features) < seq_len + label_horizon or len(stock_prices) < seq_len + label_horizon:
                continue
            
            # 创建特征日期到价格索引的映射
            price_dates = stock_prices['date'].values
            price_close = stock_prices['close'].values
            
            # 使用numpy操作进行日期匹配
            date_to_idx = {date: idx for idx, date in enumerate(price_dates)}
            
            # 批量处理每个时间窗口
            for i in range(0, len(stock_features) - seq_len - label_horizon + 1, sampling_interval):
                # 特征序列
                feature_seq = stock_features.iloc[i:i+seq_len][feature_cols].values
                
                # 标签序列 - 使用预计算的索引
                label_end_date = stock_features.iloc[i+seq_len-1]['end_date']
                if label_end_date not in date_to_idx:
                    continue
                    
                label_start_idx = date_to_idx[label_end_date]
                if label_start_idx + label_horizon >= len(price_close):
                    continue
                
                # 计算逐日收益序列
                future_prices = price_close[label_start_idx:label_start_idx+label_horizon+1]
                returns = (future_prices[1:] - future_prices[:-1]) / future_prices[:-1]
                
                features_list.append(feature_seq)
                labels_list.append(returns)
                dates_list.append(label_end_date)
        
        if not features_list:
            print("✗ 没有找到有效数据")
            return None, None, None
            
        # 批量转换为numpy数组
        print("✓ 转换为numpy数组...")
        features = np.stack(features_list)
        labels = np.stack(labels_list)
        
        print(f"✓ 最终数据形状 - 特征: {features.shape}, 标签: {labels.shape}")
        return features, labels, dates_list
    
    def create_data_loaders(self, data, batch_size=256):
        """创建数据加载器 - 优化GPU性能"""
        train_dataset = FactorDataset(
            data['train_features'], 
            data['train_labels'], 
            seq_len=40,  # 固定40个交易日
            label_horizon=10
        )
        
        val_dataset = FactorDataset(
            data['val_features'], 
            data['val_labels'], 
            seq_len=40,
            label_horizon=10
        )
        
        # 优化数据加载器配置，充分利用RTX 4090
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=8,  # 增加worker数量
            pin_memory=True,
            persistent_workers=True,  # 保持worker进程
            prefetch_factor=2  # 预取因子
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size*2, 
            shuffle=False, 
            num_workers=8,  # 增加worker数量
            pin_memory=True,
            persistent_workers=True,  # 保持worker进程
            prefetch_factor=2  # 预取因子
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch - 使用混合精度训练"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc="Training"):
            batch_features = batch_features.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                # 前向传播
                predictions = self.model(batch_features, batch_labels)
                
                # IC损失函数
                loss = ic_loss_function(predictions, batch_labels)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """验证模型 - 优化GPU性能"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                # 使用混合精度推理
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch_features, batch_labels)
                    loss = ic_loss_function(predictions, batch_labels)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        val_ic = np.corrcoef(all_predictions, all_labels)[0, 1]
        
        return avg_loss, val_ic
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """完整训练流程"""
        print("="*60)
        print("开始Transformer因子训练")
        print(f"✓ 训练轮数: {num_epochs}")
        print(f"✓ 损失函数: IC的相反数")
        print(f"✓ 模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)
        
        best_val_ic = -float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_ic = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ics.append(val_ic)
            
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证IC: {val_ic:.6f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6e}")
            
            # 保存最佳模型
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_ic': val_ic,
                    'config': self.config
                }, 'best_transformer_factor_model.pth')
                print(f"✓ 保存最佳模型 (IC: {val_ic:.6f})")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"验证IC连续{patience}轮未改善，提前停止训练")
                break
        
        print(f"\n训练完成！最佳验证IC: {best_val_ic:.6f}")
        return best_val_ic

    def train_ensemble_models(self, seeds=[42, 142, 242], save_models=True):
        """
        训练集成模型：使用不同随机数种子训练三次，等权集成
        严格按照研究规格实现
        
        Args:
            seeds: 随机数种子列表，默认[42, 142, 242]
            save_models: 是否保存训练的模型
        """
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
        data = self.load_and_process_data()
        if data is None:
            print("✗ 数据加载失败")
            return None
        
        train_loader, val_loader = self.create_data_loaders(
            data, batch_size=self.config['training_params']['batch_size']
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
            model = TransformerSeq2SeqModel(**self.config['model_params']).to(self.device)
            
            # 重新初始化优化器和调度器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['training_params']['learning_rate'],
                weight_decay=self.config['training_params']['weight_decay']
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training_params']['num_epochs']
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
                num_epochs=self.config['training_params']['num_epochs']
            )
            
            # 保存模型信息
            model_info = {
                'model_id': i + 1,
                'seed': seed,
                'best_ic': best_ic,
                'config': self.config.copy()
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
                    'config': self.config,
                    'scaler_params': {
                        'mean': data['scaler'].mean_,
                        'scale': data['scaler'].scale_
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
            'scaler': data['scaler']
        }
    
    def load_trained_models(self, model_dir='trained_models'):
        """
        加载已训练的集成模型
        
        Args:
            model_dir: 模型文件目录
            
        Returns:
            dict: 包含模型列表和相关信息
        """
        if not os.path.exists(model_dir):
            print(f"✗ 模型目录不存在: {model_dir}")
            return None
        
        # 加载集成信息
        ensemble_info_file = os.path.join(model_dir, 'ensemble_info.json')
        if not os.path.exists(ensemble_info_file):
            print(f"✗ 集成信息文件不存在: {ensemble_info_file}")
            return None
        
        with open(ensemble_info_file, 'r') as f:
            ensemble_info = json.load(f)
        
        print(f"✓ 加载集成信息: {len(ensemble_info['models'])} 个模型")
        
        # 加载模型
        models = []
        scalers = []
        
        for model_info in ensemble_info['models']:
            seed = model_info['seed']
            model_file = os.path.join(model_dir, f'transformer_model_seed_{seed}.pth')
            
            if os.path.exists(model_file):
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # 创建模型
                model = TransformerSeq2SeqModel(**checkpoint['config']['model_params'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                models.append(model)
                
                # 加载scaler参数
                if 'scaler_params' in checkpoint:
                    scaler = StandardScaler()
                    scaler.mean_ = checkpoint['scaler_params']['mean']
                    scaler.scale_ = checkpoint['scaler_params']['scale']
                    scalers.append(scaler)
                else:
                    scalers.append(None)
                
                print(f"✓ 加载模型: seed_{seed}, IC: {model_info['best_ic']:.6f}")
            else:
                print(f"✗ 模型文件不存在: {model_file}")
        
        if not models:
            print("✗ 没有成功加载任何模型")
            return None
        
        return {
            'models': models,
            'scalers': scalers,
            'ensemble_info': ensemble_info
        }
    
    def calculate_ensemble_factor_scores_optimized(self, start_date='2017-01-01', end_date='2025-02-28', 
                                                 model_dir='trained_models', save_results=True, batch_size=1000):
        """
        优化版本：计算集成因子得分 - 大幅提升运算速度
        
        优化策略:
        1. 预先构建所有历史序列数据，避免重复查询
        2. GPU加速推理
        3. 批量处理优化
        4. 内存高效的数据处理
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            model_dir: 模型目录
            save_results: 是否保存结果
            batch_size: 批量处理大小
            
        Returns:
            factor_scores_df: 因子得分数据框
        """
        print("\n" + "="*70)
        print("计算集成因子得分 - 优化版本")
        print("="*70)
        print("  性能优化:")
        print("✓ 预先构建历史序列数据")
        print("✓ GPU加速推理")
        print("✓ 批量处理优化")
        print("✓ 内存高效处理")
        print("="*70)
        
        # 加载训练好的模型和标准化器
        model_data = self.load_trained_models(model_dir)
        
        if model_data is None:
            print("✗ 没有找到训练好的模型")
            return None
        
        # 类型检查确保model_data不是None
        assert model_data is not None, "模型数据不应为None"
        models = model_data['models']
        scalers = model_data['scalers']
        
        print(f"✓ 加载了 {len(models)} 个训练模型")
        
        # 移动模型到GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i, model in enumerate(models):
            models[i] = model.to(device)
        print(f"✓ 模型已移动到: {device}")
        
        # 加载特征数据
        print("正在加载和预处理特征数据...")
        features_df = pd.read_csv('equal_volume_features_all.csv')
        features_df['end_date'] = pd.to_datetime(features_df['end_date'])
        
        # 时间过滤
        features_df = features_df[
            (features_df['end_date'] >= start_date) & 
            (features_df['end_date'] <= end_date)
        ]
        
        print(f"✓ 时间过滤后特征数据: {features_df.shape}")
        
        # 6大基本特征
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        
        # 步骤1: 预构建历史序列数据
        print("\n 步骤1: 预构建历史序列数据...")
        sequence_data = self._prebuild_sequences_optimized(features_df, feature_cols)
        
        if not sequence_data:
            print("✗ 序列数据构建失败")
            return None
        
        print(f"✓ 成功构建 {len(sequence_data)} 条序列记录")
        
        # 步骤2: 批量预测
        print(f"\n 步骤2: 批量预测 (batch_size={batch_size})...")
        factor_scores = self._batch_predict_optimized(
            sequence_data, models, scalers[0], device, batch_size
        )
        
        # 转换为DataFrame
        factor_scores_df = pd.DataFrame(factor_scores)
        
        print(f"\n 计算完成！")
        print(f"✓ 总计 {len(factor_scores_df)} 条记录")
        print(f"✓ 时间范围: {factor_scores_df['date'].min()} ~ {factor_scores_df['date'].max()}")
        print(f"✓ 股票数量: {factor_scores_df['StockID'].nunique()}")
        
        # 保存结果
        if save_results:
            os.makedirs('factor_results', exist_ok=True)
            output_file = 'factor_results/ensemble_factor_scores.csv'
            factor_scores_df.to_csv(output_file, index=False)
            print(f"✓ 因子得分已保存: {output_file}")
        
        return factor_scores_df
    
    def _prebuild_sequences_optimized(self, features_df, feature_cols, seq_len=40):
        """
        优化的序列数据预构建
        
        Args:
            features_df: 特征数据框
            feature_cols: 特征列名
            seq_len: 序列长度
            
        Returns:
            sequence_data: 预构建的序列数据列表
        """
        # 按股票分组，预先排序
        print("  正在按股票分组并排序...")
        stock_groups = {}
        for stock_id in features_df['StockID'].unique():
            stock_data = features_df[features_df['StockID'] == stock_id].sort_values('end_date')
            stock_groups[stock_id] = stock_data
        
        print(f"  ✓ 分组完成，共 {len(stock_groups)} 只股票")
        
        # 获取所有需要预测的日期
        prediction_dates = sorted(features_df['end_date'].unique())
        print(f"  ✓ 需要处理 {len(prediction_dates)} 个日期")
        
        sequence_data = []
        total_combinations = len(stock_groups) * len(prediction_dates)
        processed = 0
        
        print("  正在构建序列数据...")
        
        for date in tqdm(prediction_dates, desc="构建序列"):
            date_sequences = []
            date_stocks = []
            
            for stock_id, stock_data in stock_groups.items():
                # 获取该股票在该日期之前的历史数据
                historical_data = stock_data[stock_data['end_date'] <= date]
                
                if len(historical_data) < 20:  # 至少需要20天历史数据
                    continue
                
                # 取最近40天或所有可用数据
                recent_data = historical_data.tail(seq_len)
                seq_features = recent_data[feature_cols].values
                
                # 如果不足40天，用最早的数据填充
                if len(seq_features) < seq_len:
                    padding_size = seq_len - len(seq_features)
                    first_record = seq_features[0:1]
                    padding = np.tile(first_record, (padding_size, 1))
                    seq_features = np.vstack([padding, seq_features])
                
                date_sequences.append(seq_features)
                date_stocks.append(stock_id)
                processed += 1
            
            # 批量保存该日期的所有序列
            if date_sequences:
                batch_features = np.array(date_sequences)  # (n_stocks, seq_len, n_features)
                
                sequence_data.append({
                    'date': date,
                    'stocks': date_stocks,
                    'features': batch_features
                })
        
        print(f"  ✓ 序列构建完成，处理率: {processed/total_combinations*100:.1f}%")
        return sequence_data
    
    def _batch_predict_optimized(self, sequence_data, models, scaler, device, batch_size):
        """
        优化的批量预测
        
        Args:
            sequence_data: 预构建的序列数据
            models: 模型列表
            scaler: 标准化器
            device: 计算设备
            batch_size: 批量大小
            
        Returns:
            factor_scores: 因子得分列表
        """
        factor_scores = []
        
        for data_item in tqdm(sequence_data, desc="批量预测"):
            date = data_item['date']
            stocks = data_item['stocks']
            features = data_item['features']  # (n_stocks, seq_len, n_features)
            
            if len(stocks) < 10:  # 至少需要10只股票
                continue
            
            # 分批处理以避免GPU内存不足
            n_stocks = len(stocks)
            
            all_predictions = []
            
            for i in range(0, n_stocks, batch_size):
                end_idx = min(i + batch_size, n_stocks)
                batch_features = features[i:end_idx]  # (batch_size, seq_len, n_features)
                
                # 标准化
                if scaler is not None:
                    original_shape = batch_features.shape
                    batch_features = batch_features.reshape(-1, batch_features.shape[-1])
                    batch_features = scaler.transform(batch_features)
                    batch_features = batch_features.reshape(original_shape)
                
                # 转换为tensor并移动到GPU
                features_tensor = torch.FloatTensor(batch_features).to(device)
                
                # 集成预测 - 使用自回归seq2seq预测
                batch_predictions = []
                with torch.no_grad():
                    for model in models:
                        model.eval()
                        # 使用自回归预测生成未来10天收益序列
                        preds = self.seq2seq_autoregressive_predict(model, features_tensor, device)  # (batch_size, 10)
                        # 取首日预测作为因子得分
                        pred = preds[:, 0]  # (batch_size,) - 已经是numpy数组
                        batch_predictions.append(pred)
                
                # 等权平均
                ensemble_pred = np.mean(batch_predictions, axis=0)
                all_predictions.extend(ensemble_pred)
            
            # 保存结果
            for stock_id, prediction in zip(stocks, all_predictions):
                factor_scores.append({
                    'date': date,
                    'StockID': stock_id,
                    'factor_score': prediction
                })
        
        return factor_scores

    def seq2seq_autoregressive_predict(self, model, src, device):
        """
        自回归生成未来10天收益序列
        src: (batch, 40, 6)
        返回: (batch, 10)
        """
        model.eval()
        batch_size = src.shape[0]
        label_horizon = 10
        preds = torch.zeros((batch_size, label_horizon), device=device)
        decoder_input = torch.zeros((batch_size, 1), device=device)
        memory = model.transformer_encoder(model.pos_encoding(model.input_embedding(src) * math.sqrt(model.d_model)))
        for t in range(label_horizon):
            tgt_emb = model.output_embedding(decoder_input.unsqueeze(-1)) * math.sqrt(model.d_model)
            tgt_emb = model.decoder_pos_encoding(tgt_emb)
            tgt_emb = model.dropout(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(t+1).to(device)
            out = model.transformer_decoder(
                tgt_emb, memory, tgt_mask=tgt_mask
            )
            step_pred = model.fc_out(out[:, -1, :]).squeeze(-1)  # (batch,)
            preds[:, t] = step_pred
            decoder_input = torch.cat([decoder_input, step_pred.unsqueeze(1)], dim=1)
        return preds.detach().cpu().numpy()

    def _ensemble_predict_with_sequences(self, batch_features, models):
        """
        使用已准备好的序列数据进行集成预测（自回归seq2seq）
        Returns: (n_stocks, 10)
        """
        features_tensor = torch.FloatTensor(batch_features)
        device = next(models[0].parameters()).device
        all_model_preds = []
        with torch.no_grad():
            for model in models:
                model.eval()
                preds = self.seq2seq_autoregressive_predict(model, features_tensor.to(device), device)  # (n_stocks, 10)
                all_model_preds.append(preds)
        ensemble_pred = np.mean(all_model_preds, axis=0)  # (n_stocks, 10)
        return ensemble_pred

    def calculate_ensemble_factor_scores(self, start_date='2017-01-01', end_date='2025-02-28', 
                                       model_dir='trained_models', save_results=True):
        """
        计算集成因子得分（用于回测）
        """
        print("="*70)
        print("计算集成因子得分")
        print(f"✓ 回测区间: {start_date} ~ {end_date}")
        print(f"✓ 模型目录: {model_dir}")
        print("="*70)
        ensemble_data = self.load_trained_models(model_dir)
        if ensemble_data is None:
            return None
        models = ensemble_data['models']
        scalers = ensemble_data['scalers']
        try:
            features_df = pd.read_csv('equal_volume_features_all.csv')
            features_df['end_date'] = pd.to_datetime(features_df['end_date'])
        except:
            print("✗ 无法加载特征数据")
            return None
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        features_df = features_df[
            (features_df['end_date'] >= start_date) & 
            (features_df['end_date'] <= end_date)
        ]
        print(f"✓ 时间过滤后特征数据: {features_df.shape}")
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        factor_scores = []
        for date in sorted(features_df['end_date'].unique()):
            print(f"  处理日期: {date}")
            date_data = features_df[features_df['end_date'] == date]
            if len(date_data) < 10:
                continue
            stock_features = []
            valid_stocks = []
            for _, stock_row in date_data.iterrows():
                stock_id = stock_row['StockID']
                end_date = stock_row['end_date']
                stock_history = features_df[
                    (features_df['StockID'] == stock_id) & 
                    (features_df['end_date'] <= end_date)
                ].sort_values('end_date').tail(40)
                if len(stock_history) < 20:
                    continue
                seq_features = stock_history[feature_cols].values
                if len(seq_features) < 40:
                    padding_size = 40 - len(seq_features)
                    first_record = seq_features[0:1]
                    padding = np.tile(first_record, (padding_size, 1))
                    seq_features = np.vstack([padding, seq_features])
                stock_features.append(seq_features)
                valid_stocks.append(stock_id)
            if len(stock_features) < 10:
                continue
            batch_features = np.array(stock_features)
            if scalers[0] is not None:
                original_shape = batch_features.shape
                batch_features = batch_features.reshape(-1, batch_features.shape[-1])
                batch_features = scalers[0].transform(batch_features)
                batch_features = batch_features.reshape(original_shape)
            ensemble_predictions = self._ensemble_predict_with_sequences(batch_features, models)  # (n_stocks, 10)
            for i, stock_id in enumerate(valid_stocks):
                # 保存首日、均值、全序列
                factor_scores.append({
                    'date': date,
                    'StockID': stock_id,
                    'factor_score_first': ensemble_predictions[i, 0],
                    'factor_score_mean': ensemble_predictions[i].mean(),
                    'factor_score_seq': ','.join([f'{x:.6f}' for x in ensemble_predictions[i]])
                })
        factor_scores_df = pd.DataFrame(factor_scores)
        print(f"✓ 计算完成，总计 {len(factor_scores_df)} 条记录")
        print(f"✓ 时间范围: {factor_scores_df['date'].min()} ~ {factor_scores_df['date'].max()}")
        print(f"✓ 股票数量: {factor_scores_df['StockID'].nunique()}")
        if save_results:
            os.makedirs('factor_results', exist_ok=True)
            output_file = 'factor_results/ensemble_factor_scores.csv'
            factor_scores_df.to_csv(output_file, index=False)
            print(f"✓ 因子得分已保存: {output_file}")
        return factor_scores_df

def main():
    """主函数 - 多种子集成因子训练"""
    
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Transformer因子训练 - 多种子集成')
    parser.add_argument('--mode', choices=['train', 'backtest', 'full'], default='full',
                      help='运行模式: train(仅训练), backtest(仅回测), full(完整流程)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 142, 242],
                      help='随机数种子列表')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--fast', action='store_true',
                      help='快速训练模式：减少模型复杂度，增大batch_size')
    
    try:
        args = parser.parse_args()
    except:
        # 如果没有命令行参数，使用默认值
        class Args:
            mode = 'full'
            seeds = [42, 142, 242]
            epochs = 50
            fast = False
        args = Args()
    
    print("Transformer因子合成训练 - 多种子集成版本")
    print("="*80)
    print("技术规格:")
    print("✓ 特征X: 个股过去40个交易日的6大因子数据 (OHLC+VWAP+Amount)")
    print("✓ 标签y: 个股未来10个交易日（T+1～T+11）的收益率")
    print("✓ 训练数据: 从2013年开始，每5个交易日采样一次")
    print("✓ 数据划分: 训练集和验证集依时间先后按照4:1比例划分")
    print("✓ 损失函数: 预测值与标签之间IC的相反数")
    print("✓ 集成策略: 3个不同种子模型等权集成，减轻随机性干扰")
    print("="*80)
    print(f"运行模式: {args.mode}")
    print(f"随机数种子: {args.seeds}")
    print(f"快速模式: {'是' if args.fast else '否'}")
    print("="*80)
    
    # 模型配置 - 根据模式选择
    if args.fast:
        # 快速训练配置 - 充分利用RTX 4090
        model_config = {
            'model_params': {
                'input_dim': 6,
                'd_model': 64,            # 减少模型维度
                'nhead': 4,               # 减少注意力头数
                'num_layers': 2,          # 减少层数
                'dim_feedforward': 256,   # 减少FF维度
                'dropout': 0.1,
                'label_horizon': 10
            },
            'training_params': {
                'learning_rate': 3e-4,    # 稍微提高学习率
                'weight_decay': 1e-4,
                'batch_size': 1024,       # 大幅增大batch_size，充分利用24GB显存
                'num_epochs': 30          # 减少训练轮数
            }
        }
        print("使用快速训练配置:")
        print("✓ 模型维度: 64 (原128)")
        print("✓ 注意力头数: 4 (原8)")
        print("✓ 层数: 2 (原3)")
        print("✓ 批大小: 1024 (原256) - 充分利用RTX 4090")
        print("✓ 训练轮数: 30 (原50)")
    else:
        # 标准配置 - 优化GPU性能
        model_config = {
            'model_params': {
                'input_dim': 6,           # 6大基本特征
                'd_model': 128,           # 模型维度
                'nhead': 8,               # 多头注意力头数
                'num_layers': 3,          # Encoder层数
                'dim_feedforward': 512,   # Feed Forward维度
                'dropout': 0.1,
                'label_horizon': 10
            },
            'training_params': {
                'learning_rate': 2e-4,
                'weight_decay': 1e-4,
                'batch_size': 512,        # 增大batch_size，充分利用GPU
                'num_epochs': args.epochs
            }
        }
        print("使用标准配置:")
        print("✓ 批大小: 512 (充分利用RTX 4090)")
        print("✓ 混合精度训练: 启用")
        print("✓ CUDA优化: 启用")
    
    # 初始化训练器
    trainer = TransformerFactorTrainer(model_config)
    
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
    
    if args.mode in ['backtest', 'full']:
        print("\n" + "="*60)
        print("阶段2: 集成因子回测")
        print("="*60)
        
        # 根据模式选择回测参数
        if args.fast:
            # 快速回测配置
            backtest_batch_size = 2000  # 更大的批量
            print("使用快速回测配置:")
            print("✓ 批量大小: 2000 (充分利用RTX 4090)")
            print("✓ GPU加速: 启用")
            print("✓ 预构建序列: 启用")
        else:
            # 标准回测配置
            backtest_batch_size = 1000
            print("使用标准回测配置:")
            print("✓ 批量大小: 1000")
            print("✓ GPU加速: 启用")
            print("✓ 预构建序列: 启用")
        
        # 计算集成因子得分 - 使用优化版本
        factor_scores = trainer.calculate_ensemble_factor_scores_optimized(
            start_date='2017-01-01',
            end_date='2025-02-28',
            save_results=True,
            batch_size=backtest_batch_size  # 使用动态批量大小
        )
        
        if factor_scores is not None:
            print(f"\n✓ 回测完成！")
            print(f"✓ 因子得分已保存到 factor_results/ensemble_factor_scores.csv")
        else:
            print("✗ 回测失败")
    
    print("\n" + "="*80)
    print("训练流程完成！")
    print("="*80)

if __name__ == "__main__":
    main() 