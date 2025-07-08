#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer因子合成训练模型
基于等量K线特征进行因子训练，预测未来10日收益率

模型架构：
- Input Embedding + Positional Encoding
- Multi-Head Attention + Feed Forward
- Encoder Layers with Add & Norm
- Final Linear Layer for Prediction
- IC Loss Function
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import math
import os
from datetime import datetime, timedelta
import psutil

class FactorDataset(Dataset):
    """因子数据集类"""
    
    def __init__(self, features, labels, seq_len=40):
        """
        Args:
            features: 特征数据 (samples, features)
            labels: 标签数据 (samples,)
            seq_len: 序列长度，默认40个交易日
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.features) - self.seq_len + 1
    
    def __getitem__(self, idx):
        # 获取过去seq_len天的特征作为序列
        feature_seq = self.features[idx:idx + self.seq_len]  # (seq_len, feature_dim)
        label = self.labels[idx + self.seq_len - 1]  # 当前时点的标签
        
        return feature_seq, label

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
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    """Transformer Encoder模块"""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Input Embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 优化的Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 改为batch_first提升性能
            norm_first=True    # 预归一化，更稳定更快
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False  # 禁用嵌套张量优化兼容性
        )
        
        # Output Layer
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: (batch_size, seq_len, input_dim)
            src_mask: 可选的掩码
        
        Returns:
            output: (batch_size,) 预测的收益率
        """
        # 现在使用batch_first=True，不需要转换维度
        # src: (batch_size, seq_len, input_dim)
        
        # Input Embedding
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Positional Encoding - 适配batch_first
        batch_size, seq_len, _ = src.shape
        # self.pos_encoding.pe的形状是 (max_len, 1, d_model)
        # 我们需要 (batch_size, seq_len, d_model)
        pos_encoding = self.pos_encoding.pe[:seq_len, 0, :].unsqueeze(0).expand(batch_size, -1, -1)
        src = src + pos_encoding
        src = self.dropout(src)
        
        # Transformer Encoder (batch_first=True)
        output = self.transformer_encoder(src, src_mask)
        
        # 使用最后一个时间步的输出进行预测
        # output: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        last_output = output[:, -1, :]
        
        # Final prediction
        prediction = self.output_projection(last_output)  # (batch_size, 1)
        
        return prediction.squeeze(-1)  # (batch_size,)

def ic_loss(predictions, targets):
    """
    IC损失函数：预测值与真实值之间IC的相反数
    
    Args:
        predictions: 预测值 (batch_size,)
        targets: 真实值 (batch_size,)
    
    Returns:
        loss: -IC值
    """
    # 计算皮尔逊相关系数
    pred_mean = torch.mean(predictions)
    target_mean = torch.mean(targets)
    
    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean
    
    numerator = torch.sum(pred_centered * target_centered)
    pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
    target_std = torch.sqrt(torch.sum(target_centered ** 2))
    
    # 避免除零
    denominator = pred_std * target_std + 1e-8
    
    ic = numerator / denominator
    
    # 返回IC的相反数作为损失
    return -ic

def rank_ic_loss(predictions, targets):
    """
    Rank IC损失函数：基于排序的IC计算，对异常值更稳健
    
    Args:
        predictions: 预测值 (batch_size,)
        targets: 真实值 (batch_size,)
    
    Returns:
        loss: -Rank IC值
    """
    # 计算排序
    pred_ranks = torch.argsort(torch.argsort(predictions).float())
    target_ranks = torch.argsort(torch.argsort(targets).float())
    
    # 计算Spearman相关系数（基于排序的相关）
    pred_mean = torch.mean(pred_ranks)
    target_mean = torch.mean(target_ranks)
    
    pred_centered = pred_ranks - pred_mean
    target_centered = target_ranks - target_mean
    
    numerator = torch.sum(pred_centered * target_centered)
    pred_std = torch.sqrt(torch.sum(pred_centered ** 2))
    target_std = torch.sqrt(torch.sum(target_centered ** 2))
    
    denominator = pred_std * target_std + 1e-8
    rank_ic = numerator / denominator
    
    return -rank_ic

def mse_loss(predictions, targets):
    """
    均方误差损失：最基础但稳定的损失函数
    """
    return torch.mean((predictions - targets) ** 2)

def huber_loss(predictions, targets, delta=1.0):
    """
    Huber损失：对异常值更稳健的损失函数
    """
    diff = torch.abs(predictions - targets)
    return torch.where(diff < delta, 
                      0.5 * diff ** 2, 
                      delta * (diff - 0.5 * delta)).mean()

def combined_loss(predictions, targets, alpha=0.7, beta=0.3):
    """
    组合损失：结合IC损失和MSE损失
    
    Args:
        alpha: IC损失权重
        beta: MSE损失权重
    """
    ic_l = ic_loss(predictions, targets)
    mse_l = mse_loss(predictions, targets)
    
    # 归一化MSE损失到合适的尺度
    mse_normalized = mse_l / (torch.var(targets) + 1e-8)
    
    return alpha * ic_l + beta * mse_normalized

def directional_loss(predictions, targets):
    """
    方向损失：关注预测方向的正确性
    """
    # 将连续值转换为方向（正/负/零）
    pred_direction = torch.sign(predictions)
    target_direction = torch.sign(targets)
    
    # 计算方向一致性
    correct_direction = (pred_direction * target_direction > 0).float()
    
    # 返回方向错误率
    return 1.0 - torch.mean(correct_direction)

class FactorTrainer:
    """因子训练器"""
    
    def __init__(self, model_config):
        """
        初始化训练器
        
        Args:
            model_config: 模型配置字典
        """
        self.config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 损失函数选择
        self.loss_type = model_config.get('loss_type', 'combined')  # 默认使用组合损失
        self.loss_functions = {
            'ic': ic_loss,
            'rank_ic': rank_ic_loss,
            'mse': mse_loss,
            'huber': huber_loss,
            'combined': combined_loss,
            'directional': directional_loss
        }
        
        # 显示详细的GPU信息
        self._print_device_info()
        
        # 初始化模型
        self.model = TransformerEncoder(**model_config['model_params']).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=model_config['learning_rate'],
            weight_decay=model_config.get('weight_decay', 1e-5)
        )
        
        # 优化的学习率调度器 - 余弦退火+重启
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 混合精度训练 - 只在GPU上启用
        self.use_amp = self.device.type == 'cuda' and model_config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()
            print("启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            if self.device.type == 'cuda':
                print("混合精度训练已禁用")
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_ics = []
        
    def _print_device_info(self):
        """打印设备信息"""
        print("=" * 60)
        print("GPU加速信息")
        print("=" * 60)
        
        if torch.cuda.is_available():
            print(f"CUDA可用")
            print(f"PyTorch版本: {torch.__version__}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
            
            # 当前GPU信息
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            print(f"使用GPU: {gpu_name}")
            print(f"GPU显存: {gpu_memory:.1f} GB")
            print(f"设备索引: cuda:{current_device}")
            
            # GPU内存使用情况
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            print(f"已分配显存: {memory_allocated:.2f} GB")
            print(f"已预留显存: {memory_reserved:.2f} GB")
            
        else:
            print("CUDA不可用，使用CPU训练")
            print(f"CPU核心数: {psutil.cpu_count()}")
            print(f"可用内存: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
        print("=" * 60)
    
    def _monitor_gpu_memory(self):
        """监控GPU内存使用"""
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            print(f"GPU内存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB / {memory_total:.1f}GB")
    
    def _adjust_loss_function(self, epoch, current_ic):
        """
        根据训练情况动态调整损失函数
        
        Args:
            epoch: 当前训练轮次
            current_ic: 当前IC值
        """
        # 如果连续多轮IC为负，考虑切换损失函数
        if len(self.val_ics) >= 5:
            recent_ics = self.val_ics[-5:]
            avg_recent_ic = sum(recent_ics) / len(recent_ics)
            
            # 如果最近5轮平均IC都小于-0.01，且当前使用IC损失
            if avg_recent_ic < -0.01 and self.loss_type == 'ic':
                print(f"\n检测到IC持续为负 (最近5轮平均: {avg_recent_ic:.4f})")
                print("自动切换到组合损失函数以改善训练稳定性")
                self.loss_type = 'combined'
                return True
            
            # 如果使用组合损失后IC仍然很差，切换到MSE
            elif avg_recent_ic < -0.02 and self.loss_type == 'combined':
                print(f"\n组合损失效果不佳 (最近5轮平均: {avg_recent_ic:.4f})")
                print("切换到MSE损失函数专注于预测准确性")
                self.loss_type = 'mse'
                return True
            
            # 如果MSE也不行，尝试Huber损失
            elif avg_recent_ic < -0.015 and self.loss_type == 'mse' and epoch > 15:
                print(f"\nMSE损失效果不佳 (最近5轮平均: {avg_recent_ic:.4f})")
                print("切换到Huber损失函数处理异常值")
                self.loss_type = 'huber'
                return True
        
        return False
    
    def _check_and_fix_labels(self, predictions, targets):
        """
        检查并修正标签问题
        
        Args:
            predictions: 模型预测值
            targets: 真实标签
        
        Returns:
            corrected_targets: 修正后的标签
        """
        # 计算当前IC
        current_ic = np.corrcoef(predictions.detach().cpu().numpy(), 
                               targets.detach().cpu().numpy())[0, 1]
        
        # 如果IC明显为负且接近-1，可能标签反向了
        if current_ic < -0.5:
            print(f"\n警告：检测到强负相关 (IC={current_ic:.4f})")
            print("可能存在标签方向错误，建议检查收益率计算逻辑")
            
            # 可选：自动反转标签（谨慎使用）
            # return -targets
        
        return targets
        
    def prepare_data(self, equal_volume_file='equal_volume_features_all.csv', 
                    stock_data_file='stock_price_vol_d.txt'):
        """
        准备训练数据
        
        Args:
            equal_volume_file: 等量K线特征文件
            stock_data_file: 原始股票数据文件
        
        Returns:
            train_loader, val_loader: 训练和验证数据加载器
        """
        print("正在准备训练数据...")
        
        # 读取等量K线特征
        try:
            features_df = pd.read_csv(equal_volume_file)
            print(f"等量K线特征数据形状: {features_df.shape}")
        except FileNotFoundError:
            print(f"未找到特征文件 {equal_volume_file}，请先运行等量K线构建脚本")
            return None, None
        
        # 读取原始股票数据用于计算收益率标签
        try:
            stock_data = pd.read_feather(stock_data_file)
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            print(f"原始股票数据形状: {stock_data.shape}")
        except FileNotFoundError:
            print(f"未找到股票数据文件 {stock_data_file}")
            return None, None
        
        # 数据预处理
        processed_data = self._process_features_and_labels(features_df, stock_data)
        
        if processed_data is None:
            return None, None
        
        features, labels = processed_data
        
        # 数据标准化（按特征维度标准化）
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 检查标签分布
        print(f"标签统计信息:")
        print(f"  均值: {labels.mean():.6f}")
        print(f"  标准差: {labels.std():.6f}")
        print(f"  最小值: {labels.min():.6f}")
        print(f"  最大值: {labels.max():.6f}")
        print(f"  正收益率占比: {(labels > 0).mean():.4f}")
        
        # 创建数据集
        dataset = FactorDataset(features_scaled, labels, seq_len=self.config['seq_len'])
        
        # 按时间序列划分训练集和验证集（避免数据泄露）
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        # 高性能数据加载器设置
        num_workers = min(8, os.cpu_count() or 1) if torch.cuda.is_available() else 2
        pin_memory = torch.cuda.is_available()
        
        print(f"数据加载器配置: num_workers={num_workers}, pin_memory={pin_memory}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,  # 适当shuffle提升泛化能力
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=4,  # 预取更多批次
            drop_last=True  # 保证批次大小一致
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'] * 2,  # 验证时用更大批次
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _process_features_and_labels(self, features_df, stock_data):
        """
        处理特征和标签数据
        
        Args:
            features_df: 等量K线特征数据
            stock_data: 原始股票数据
        
        Returns:
            features, labels: 处理后的特征和标签数组
        """
        print("正在处理特征和标签...")
        
        # 选择6大基本特征
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        
        # 检查特征列是否存在
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            print(f"缺少特征列: {missing_cols}")
            return None
        
        # 构建特征矩阵
        features_list = []
        labels_list = []
        
        # 预筛选有效股票，减少循环次数
        stock_counts_features = features_df['StockID'].value_counts()
        stock_counts_prices = stock_data['StockID'].value_counts()
        valid_stocks = []
        
        for stock_id in stock_counts_features.index:
            if (stock_counts_features[stock_id] >= 50 and 
                stock_counts_prices.get(stock_id, 0) >= 50):
                valid_stocks.append(stock_id)
        
        # 限制处理的股票数量以加速训练
        valid_stocks = valid_stocks[:500]  # 最多处理500只股票
        print(f"有效股票数量: {len(valid_stocks)}")
        
        # 按股票分组处理（使用进度条）
        for stock_id in tqdm(valid_stocks, desc="处理股票数据"):
            stock_features = features_df[features_df['StockID'] == stock_id].copy()
            stock_prices = stock_data[stock_data['StockID'] == stock_id].copy()
                
            # 按日期排序
            stock_features = stock_features.sort_values('end_date')
            stock_prices = stock_prices.sort_values('date')
            
            # 计算未来10日收益率标签
            stock_prices['future_close'] = stock_prices['close'].shift(-10)
            stock_prices['future_return'] = (stock_prices['future_close'] - stock_prices['close']) / stock_prices['close']
            
            # 对齐特征和标签的日期
            stock_features['date'] = pd.to_datetime(stock_features['end_date'])
            stock_prices['date'] = pd.to_datetime(stock_prices['date'])
            
            # 合并数据
            merged_data = pd.merge(
                stock_features[feature_cols + ['date']],
                stock_prices[['date', 'future_return']],
                on='date',
                how='inner'
            )
            
            # 去除缺失值
            merged_data = merged_data.dropna()
            
            if len(merged_data) < 40:  # 至少需要40个样本
                continue
            
            # 添加到列表
            features_array = merged_data[feature_cols].values
            labels_array = merged_data['future_return'].values
            
            features_list.append(features_array)
            labels_list.append(labels_array)
        
        if not features_list:
            print("没有找到有效的特征和标签数据")
            return None
        
        # 合并所有股票的数据
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        
        print(f"最终特征矩阵形状: {features.shape}")
        print(f"最终标签数组形状: {labels.shape}")
        
        return features, labels
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc="Training"):
            batch_features = batch_features.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # 使用混合精度训练
            if self.use_amp:
                with autocast():
            predictions = self.model(batch_features)
                    loss = self.loss_functions[self.loss_type](predictions, batch_labels)
                
                # 缩放损失并反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准精度训练
                predictions = self.model(batch_features)
                loss = self.loss_functions[self.loss_type](predictions, batch_labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                # 使用混合精度推理
                if self.use_amp:
                    with autocast():
                        predictions = self.model(batch_features)
                        loss = self.loss_functions[self.loss_type](predictions, batch_labels)
                else:
                predictions = self.model(batch_features)
                    loss = self.loss_functions[self.loss_type](predictions, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # 计算验证集IC
        val_ic = np.corrcoef(all_predictions, all_labels)[0, 1]
        
        return avg_loss, val_ic
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """完整训练流程"""
        print(f"开始训练，总共 {num_epochs} 个epochs")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 显示初始GPU内存使用情况
        if torch.cuda.is_available():
            self._monitor_gpu_memory()
        
        best_val_ic = -float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 优化验证频率：前期每轮验证，后期每2轮验证一次
            should_validate = epoch < 20 or epoch % 2 == 0
            
            if should_validate:
            # 验证
            val_loss, val_ic = self.validate(val_loader)
            else:
                # 跳过验证，使用上一次的值
                val_loss, val_ic = self.val_losses[-1] if self.val_losses else 0.0, self.val_ics[-1] if self.val_ics else 0.0
            
            # 更新学习率（余弦退火调度器）
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ics.append(val_ic)
            
            # 显示训练结果
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证IC: {val_ic:.6f}")
            print(f"学习率: {current_lr:.2e}")
            print(f"当前损失函数: {self.loss_type}")
            
            if new_lr != current_lr:
                print(f"学习率降低至: {new_lr:.2e}")
            
            # 动态调整损失函数
            if self._adjust_loss_function(epoch, val_ic):
                print(f"已切换损失函数至: {self.loss_type}")
                # 重置早停计数器，给新损失函数一些机会
                patience_counter = max(0, patience_counter - 3)
            
            # 显示GPU内存使用情况
            if torch.cuda.is_available() and epoch % 5 == 0:
                self._monitor_gpu_memory()
            
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
                }, 'best_transformer_model.pth')
                print(f"保存最佳模型 (IC: {val_ic:.6f})")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"验证IC连续{patience}轮未改善，提前停止训练")
                break
        
        print(f"\n训练完成！最佳验证IC: {best_val_ic:.6f}")
        
        # 最终GPU内存使用情况
        if torch.cuda.is_available():
            print("\n最终GPU内存使用情况:")
            self._monitor_gpu_memory()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return best_val_ic
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True)
        
        # IC曲线
        ax2.plot(self.val_ics, label='验证IC', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IC')
        ax2.set_title('验证IC变化')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("训练曲线已保存为 training_curves.png")

def main():
    """主函数"""
    print("Transformer因子合成训练")
    print("=" * 50)
    
    # 模型配置
    model_config = {
        'model_params': {
            'input_dim': 6,        # 6大基本特征
            'd_model': 128,        # 模型维度
            'nhead': 8,            # 注意力头数
            'num_layers': 4,       # Encoder层数
            'dim_feedforward': 512, # 前馈网络维度
            'dropout': 0.1         # Dropout率
        },
        'seq_len': 40,             # 序列长度（过去40个交易日）
        'batch_size': 64,          # 批大小
        'learning_rate': 1e-4,     # 学习率
        'weight_decay': 1e-5       # 权重衰减
    }
    
    print("模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # 创建训练器
    trainer = FactorTrainer(model_config)
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data()
    
    if train_loader is None or val_loader is None:
        print("数据准备失败，请检查数据文件")
        return
    
    # 开始训练
    best_ic = trainer.train(train_loader, val_loader, num_epochs=100)
    
    print(f"训练完成！最佳IC: {best_ic:.6f}")
    print("模型已保存为 best_transformer_model.pth")
    
    return trainer

if __name__ == "__main__":
    trainer = main() 