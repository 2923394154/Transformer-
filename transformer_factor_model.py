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
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch_size, d_model)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
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
            output: (batch_size, 1) 预测的收益率
        """
        # 转换维度: (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)
        src = src.transpose(0, 1)
        
        # Input Embedding
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Positional Encoding
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        # Transformer Encoder
        output = self.transformer_encoder(src, src_mask)
        
        # 使用最后一个时间步的输出进行预测
        # output: (seq_len, batch_size, d_model) -> (batch_size, d_model)
        last_output = output[-1]
        
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
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = TransformerEncoder(**model_config['model_params']).to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=model_config['learning_rate'],
            weight_decay=model_config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=5, verbose=True
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_ics = []
        
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
        
        # 数据标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 创建数据集
        dataset = FactorDataset(features_scaled, labels, seq_len=self.config['seq_len'])
        
        # 按时间序列划分训练集和验证集（避免数据泄露）
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,  # 时间序列数据不shuffle
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2
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
        
        # 按股票分组处理
        for stock_id in features_df['StockID'].unique():
            stock_features = features_df[features_df['StockID'] == stock_id].copy()
            stock_prices = stock_data[stock_data['StockID'] == stock_id].copy()
            
            if len(stock_features) < 50 or len(stock_prices) < 50:  # 数据量太少
                continue
                
            # 按日期排序
            stock_features = stock_features.sort_values('end_date')
            stock_prices = stock_prices.sort_values('date')
            
            # 计算未来10日收益率标签
            stock_prices['future_return'] = stock_prices['close'].pct_change(10).shift(-10)
            
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
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(batch_features)
            
            # 计算IC损失
            loss = ic_loss(predictions, batch_labels)
            
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
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                predictions = self.model(batch_features)
                loss = ic_loss(predictions, batch_labels)
                
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
        
        best_val_ic = -float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_ic = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_ics.append(val_ic)
            
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证IC: {val_ic:.6f}")
            
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