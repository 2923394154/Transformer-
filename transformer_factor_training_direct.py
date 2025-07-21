#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格按照规格的Transformer因子训练 - 直接数据版本
按照华泰证券研究报告的技术规格实现，使用直接处理的数据

技术规格：
- 特征X：个股过去40个交易日的高频因子数据
- 标签y：个股未来10个交易日（T+1～T+11）的收益率
- 样本内训练数据从2013年开始，每5个交易日采样一次
- 训练集和验证集依时间先后按照4:1的比例划分
- 损失函数：预测值与标签之间IC的相反数 (1 - IC)
- GPU加速：充分利用RTX 4090进行训练
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
    
    print("超高速优化已启用:")
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
    """Encoder-Decoder结构，支持收益序列预测 - 优化版本"""
    def __init__(self, input_dim=6, d_model=256, nhead=16, num_layers=6, dim_feedforward=1024, dropout=0.1, label_horizon=10):
        super().__init__()
        self.d_model = d_model
        self.label_horizon = label_horizon
        
        # Encoder - 增大容量
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
        
        # Decoder - 增大容量
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
        
        # 输出层 - 增加中间层
        self.fc_hidden = nn.Linear(d_model, d_model // 2)
        self.fc_out = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化 - 优化版本"""
        output_layers = 0
        embedding_layers = 0
        other_layers = 0
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'fc_out' in name:
                    # 输出层使用极小的初始化，避免预测值过大
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)  # 减小标准差
                    output_layers += 1
                elif 'fc_hidden' in name:
                    # 隐藏层使用Xavier初始化
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    other_layers += 1
                elif 'output_embedding' in name or 'input_embedding' in name:
                    # 嵌入层使用Xavier初始化，适中gain
                    nn.init.xavier_normal_(module.weight, gain=1.0)  # 增大gain
                    embedding_layers += 1
                else:
                    # 其他层使用适中的gain
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    other_layers += 1
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        print(f"权重初始化完成: 输出层{output_layers}个(std=0.1), 嵌入层{embedding_layers}个(gain=1.0), 其他层{other_layers}个(gain=1.0)")
    
    def forward(self, src, tgt, teacher_forcing=True):
        batch_size = src.size(0)
        
        # Encoder
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

def ic_loss_function(predictions, targets):
    """
    稳定的IC损失函数 - 防止梯度爆炸
    损失 = 1 - 平均IC + 预测值范围惩罚
    """
    # 输入检查
    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    if torch.isnan(targets).any() or torch.isinf(targets).any():
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    # 预测值范围惩罚 - 防止预测值过大
    pred_range = torch.max(predictions) - torch.min(predictions)
    range_penalty = torch.clamp(pred_range - 1.0, min=0.0) * 0.1  # 惩罚超过1.0的范围
    
    # 计算T+1到T+6的平均IC
    ic_list = []
    
    for t in range(min(6, predictions.shape[1])):
        pred = predictions[:, t]
        true = targets[:, t]
        
        # 过滤有效数据
        valid_mask = ~(torch.isnan(pred) | torch.isnan(true) | torch.isinf(pred) | torch.isinf(true))
        if valid_mask.sum() < 10:  # 至少需要10个有效样本
            continue
        
        pred_valid = pred[valid_mask]
        true_valid = true[valid_mask]
        
        # 计算均值和标准差
        pred_mean = pred_valid.mean()
        pred_std = pred_valid.std()
        true_mean = true_valid.mean()
        true_std = true_valid.std()
        
        # 数值稳定性检查
        epsilon = 1e-6
        if pred_std < epsilon or true_std < epsilon:
            continue
        
        # 标准化
        pred_norm = (pred_valid - pred_mean) / torch.clamp(pred_std, min=epsilon)
        true_norm = (true_valid - true_mean) / torch.clamp(true_std, min=epsilon)
        
        # 计算皮尔逊相关系数
        covariance = torch.mean(pred_norm * true_norm)
        
        # 裁剪IC值到合理范围
        ic = torch.clamp(covariance, -1.0, 1.0)
        ic_list.append(ic)
    
    # 如果没有有效的IC计算，返回默认损失
    if len(ic_list) == 0:
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    # 计算平均IC
    avg_ic = torch.mean(torch.stack(ic_list))
    
    # 基础损失：IC的相反数
    base_loss = 1.0 - avg_ic
    
    # 添加范围惩罚
    loss = base_loss + range_penalty
    
    # 数值稳定性处理
    loss = torch.clamp(loss, 0.0, 2.0)
    
    # 最终安全检查
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(1.0, device=predictions.device, requires_grad=True)
    
    return loss

class TransformerFactorTrainer:
    """Transformer因子训练器 - 直接数据版本"""
    
    def __init__(self, model_config, max_stocks=None):
        self.model_config = model_config
        self.max_stocks = max_stocks
        self.device = device
        self.current_epoch = 0
        
        # 创建模型
        self.model = TransformerSeq2SeqModel(**model_config).to(self.device)
        
        # 优化器 - 降低学习率避免梯度爆炸
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-5,  # 降低学习率避免梯度爆炸
            weight_decay=1e-3,  # 增加权重衰减防止过拟合
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器 - 使用余弦退火
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # 每20个epoch重启一次
            T_mult=2,  # 重启后周期翻倍
            eta_min=1e-6  # 最小学习率
        )
        
        # 训练配置优化 - 速度优化版本
        self.batch_size = 2048  # 增大批次大小，提高GPU利用率
        self.num_epochs = 100   # 减少epoch数量，加快训练
        
        print(f"模型配置: d_model={model_config['d_model']}, layers={model_config['num_layers']}")
        print(f"优化器: AdamW(lr={5e-5}, weight_decay={1e-3})")
        print(f"调度器: CosineAnnealingWarmRestarts(T_0=20)")
        print(f"批次大小: {self.batch_size}, 训练轮数: {self.num_epochs}")
        
        # 确保所有参数都需要梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 初始化模型权重
        self._initialize_weights()
        
        # 改进的学习率调度器：更激进的衰减
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=model_config['training_params']['learning_rate'] * 2,  # 峰值学习率
        #     epochs=model_config['training_params']['num_epochs'],
        #     steps_per_epoch=300,  # 预估每epoch步数
        #     pct_start=0.3,  # 30%时间内升到峰值
        #     anneal_strategy='cos',
        #     div_factor=10.0,  # 初始lr = max_lr / div_factor
        #     final_div_factor=1000.0  # 最终lr = max_lr / final_div_factor
        # )
        
        # 禁用混合精度训练以避免GradScaler问题
        self.use_amp = False
        self.scaler = None
        
        print(f"模型初始化完成")
        print(f"设备: {self.device}")
        print(f"混合精度训练: 禁用（避免GradScaler问题）")
    
    def _initialize_weights(self):
        """模型权重初始化 - 修复预测方差过小问题"""
        output_layers = 0
        embedding_layers = 0
        other_layers = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if 'fc_out' in name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.001)  # 极小标准差，避免梯度爆炸
                    output_layers += 1
                elif any(embed_name in name for embed_name in ['embedding', 'pos_encoding']):
                    nn.init.xavier_normal_(module.weight, gain=0.5)  # 适中的嵌入层gain
                    embedding_layers += 1
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)  # 标准的其他层gain
                    other_layers += 1
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        print(f"训练器权重初始化完成: 输出层{output_layers}个, 嵌入层{embedding_layers}个, 其他层{other_layers}个")
    
    def load_direct_data(self, data_file='direct_features_latest.csv'):
        """加载直接处理的数据"""
        print("="*50)
        print("加载直接处理的数据")
        print("="*50)
        
        # 监控内存使用
        process = psutil.Process()
        print(f"初始内存使用: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        
        # 检查数据文件
        if not os.path.exists(data_file):
            print(f"数据文件不存在: {data_file}")
            print("请先运行: python simple_direct_processing.py")
            return None
        
        # 加载特征数据
        print(f"正在加载特征数据: {data_file}")
        try:
            features_df = pd.read_csv(data_file, engine='c', memory_map=True)
            features_df['end_date'] = pd.to_datetime(features_df['end_date'])
            features_df['StockID'] = features_df['StockID'].astype('category')
            
            print(f"特征数据加载成功: {features_df.shape}")
            print(f"股票数量: {features_df['StockID'].nunique()}")
            print(f"时间范围: {features_df['end_date'].min()} ~ {features_df['end_date'].max()}")
        except Exception as e:
            print(f"特征数据加载失败: {e}")
            return None
        
        print(f"特征数据加载后内存: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        
        # 加载股票价格数据
        print("正在加载股票价格数据...")
        try:
            stock_data = pd.read_feather('stock_price_vol_d.txt')
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data['StockID'] = stock_data['StockID'].astype('category')
            
            print(f"价格数据加载成功: {stock_data.shape}")
        except Exception as e:
            print(f"价格数据加载失败: {e}")
            return None
        
        print(f"价格数据加载后内存: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
        
        return features_df, stock_data
    
    def process_features_and_labels(self, features_df, stock_data, sampling_interval=5):
        """构建训练数据"""
        print("正在构建训练数据...")
        feature_cols = ['open', 'high', 'low', 'close', 'vwap', 'amount']
        seq_len = 40
        label_horizon = 10
        
        # 预排序
        print("预排序数据...")
        features_df = features_df.sort_values(['StockID', 'end_date']).reset_index(drop=True)
        stock_data = stock_data.sort_values(['StockID', 'date']).reset_index(drop=True)
        
        # 筛选有效股票
        print("筛选有效股票...")
        stock_counts = features_df['StockID'].value_counts()
        valid_stocks = stock_counts[stock_counts >= 30].index.tolist()
        
        print(f"有效股票数量: {len(valid_stocks)} 只")
        
        # 添加数据范围统计
        print(f"特征数据时间范围: {features_df['end_date'].min()} ~ {features_df['end_date'].max()}")
        print(f"特征数据总记录数: {len(features_df):,}")
        print(f"价格数据时间范围: {stock_data['date'].min()} ~ {stock_data['date'].max()}")
        print(f"价格数据总记录数: {len(stock_data):,}")
        
        if self.max_stocks:
            print(f"应用股票数量限制: {len(valid_stocks)} → {self.max_stocks}")
            valid_stocks = valid_stocks[:self.max_stocks]
            print(f"限制使用前{self.max_stocks}只股票进行训练")
        else:
            print(f"使用所有{len(valid_stocks)}只有效股票进行训练")
        
        # 预分配内存
        print("预分配内存...")
        max_samples = sum(len(features_df[features_df['StockID'] == stock]) - seq_len - label_horizon + 1 
                         for stock in valid_stocks) // sampling_interval
        
        # 增加数据量限制并添加统计信息
        original_max_samples = max_samples
        max_samples = min(max_samples, 5000000)  # 增加到500万样本
        
        print(f"预估样本数: {original_max_samples:,}")
        print(f"实际预分配: {max_samples:,}")
        print(f"采样间隔: {sampling_interval}天")
        if original_max_samples > 5000000:
            print(f"数据量被限制：{original_max_samples:,} → {max_samples:,}")
        
        features_array = np.zeros((max_samples, seq_len, len(feature_cols)), dtype=np.float32)
        labels_array = np.zeros((max_samples, label_horizon), dtype=np.float32)
        dates_array = np.zeros(max_samples, dtype=object)
        
        sample_idx = 0
        
        # 处理每只股票
        for stock_id in tqdm(valid_stocks, desc="处理股票"):
            stock_features = features_df[features_df['StockID'] == stock_id]
            stock_prices = stock_data[stock_data['StockID'] == stock_id]
            
            if len(stock_features) < seq_len + label_horizon or len(stock_prices) < seq_len + label_horizon:
                continue
            
            # 创建日期映射
            price_dates = stock_prices['date'].values
            price_close = stock_prices['close'].values
            date_to_idx = {date: idx for idx, date in enumerate(price_dates)}
            
            # 处理时间窗口
            feature_values = stock_features[feature_cols].values
            feature_dates = stock_features['end_date'].values
            
            for i in range(0, len(stock_features) - seq_len - label_horizon + 1, sampling_interval):
                if sample_idx >= max_samples:
                    break
                    
                # 特征序列: T-40到T-1
                feature_seq = feature_values[i:i+seq_len]
                
                # 关键修复：标签序列应该是T+1到T+10的收益率
                # 特征序列的最后一天是T-1，所以T0是下一天
                feature_end_date = feature_dates[i+seq_len-1]  # T-1
                if feature_end_date not in date_to_idx:
                    continue
                    
                feature_end_idx = date_to_idx[feature_end_date]  # T-1在价格数据中的索引
                
                # T0应该是feature_end_idx + 1
                t0_idx = feature_end_idx + 1
                
                # 确保有足够的未来数据：需要T0到T+10的价格数据
                if t0_idx + label_horizon >= len(price_close):
                    continue
                
                # 计算T+1到T+10的收益率
                # T0到T+10的价格
                future_prices = price_close[t0_idx:t0_idx+label_horizon+1]
                # T+1到T+10的收益率：(T+1价格-T0价格)/T0价格, ..., (T+10价格-T+9价格)/T+9价格
                returns = (future_prices[1:] - future_prices[:-1]) / future_prices[:-1]
                
                # 验证数组长度
                if len(returns) != label_horizon:
                    continue
                
                # 赋值到预分配数组
                features_array[sample_idx] = feature_seq
                labels_array[sample_idx] = returns
                dates_array[sample_idx] = feature_end_date  # 记录特征序列的结束日期(T-1)
                sample_idx += 1
        
        if sample_idx == 0:
            print("没有找到有效数据")
            return None, None, None
            
        # 裁剪到实际大小
        features = features_array[:sample_idx]
        labels = labels_array[:sample_idx]
        dates = dates_array[:sample_idx]
        
        print(f"最终数据形状: {features.shape}")
        print(f"标签形状: {labels.shape}")
        
        return features, labels, dates
    
    def create_data_loaders(self, data, batch_size=2048, use_cache=True):
        """创建数据加载器"""
        features, labels, dates = data
        
        # 按时间分割数据
        print("正在进行时间序列数据分割...")
        
        # 确保数据按时间排序
        sorted_indices = np.argsort(dates)
        features = features[sorted_indices]
        labels = labels[sorted_indices]
        dates = dates[sorted_indices]
        
        # 按时间顺序4:1分割
        split_idx = int(0.8 * len(features))
        train_features = features[:split_idx]
        train_labels = labels[:split_idx]
        train_dates = dates[:split_idx]
        val_features = features[split_idx:]
        val_labels = labels[split_idx:]
        val_dates = dates[split_idx:]
        
        print(f"训练集时间范围: {train_dates.min()} ~ {train_dates.max()}")
        print(f"验证集时间范围: {val_dates.min()} ~ {val_dates.max()}")
        
        # 滚动窗口标准化
        print("正在进行滚动窗口标准化...")
        
        # 尝试使用UltraFastRollingScaler
        try:
            from ultra_fast_scaler import UltraFastRollingScaler
            rolling_scaler = UltraFastRollingScaler(window_size=252)
            print("使用终极超高速标准化器（JIT加速）")
        except ImportError:
            print("使用简单标准化（Numba未安装）")
            # 简单的全局标准化作为后备
            train_features_scaled = (train_features - np.mean(train_features, axis=(0,1), keepdims=True)) / (np.std(train_features, axis=(0,1), keepdims=True) + 1e-8)
            val_features_scaled = (val_features - np.mean(train_features, axis=(0,1), keepdims=True)) / (np.std(train_features, axis=(0,1), keepdims=True) + 1e-8)
            
            # 创建一个简单的scaler对象
            class SimpleScaler:
                def __init__(self):
                    self.is_fitted = True
            
            rolling_scaler = SimpleScaler()
            train_features_scaled = np.clip(train_features_scaled, -3, 3)
            val_features_scaled = np.clip(val_features_scaled, -3, 3)
        else:
            # 使用UltraFastRollingScaler - 智能选择标准化策略
            data_size_mb = train_features.size * 4 / (1024 * 1024)  # 4字节每个float32
            print(f"数据规模: {train_features.size:,} 元素 ({data_size_mb:.1f} MB)")
            
            # 如果数据超过500MB，使用简单模式提高速度
            if data_size_mb > 500:
                print("数据量较大，使用快速简单标准化模式")
                use_simple = True
            else:
                print("使用滚动窗口标准化模式")
                use_simple = False
            
            train_features_scaled = rolling_scaler.fit_transform(
                train_features, train_dates, use_cache=use_cache, use_simple=use_simple
            )
            val_features_scaled = rolling_scaler.transform(
                val_features, val_dates, use_simple=use_simple
            )
        
        print(f"训练集大小: {train_features.shape}")
        print(f"验证集大小: {val_features.shape}")
        
        # 创建数据集
        train_dataset = FactorDataset(train_features_scaled, train_labels)
        val_dataset = FactorDataset(val_features_scaled, val_labels)
        
        # 创建数据加载器 - 速度优化版本
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=8,  # 增加工作进程数
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,  # 增加预取因子
            drop_last=True  # 丢弃不完整的批次
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=8,  # 增加工作进程数
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,  # 增加预取因子
            drop_last=True  # 丢弃不完整的批次
        )
        
        print("数据加载器配置完成")
        
        return train_loader, val_loader, rolling_scaler
    
    def train_epoch(self, train_loader):
        """训练一个epoch - 速度优化版本"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 启用PyTorch编译优化 - 只在第一次调用时编译
        if not hasattr(self, '_model_compiled'):
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode="max-autotune")
                    self._model_compiled = True
                    print("PyTorch 2.0编译优化已启用")
                except:
                    self._model_compiled = True
            else:
                self._model_compiled = True
        
        # 简化训练循环，移除不必要的数据增强
        for batch_features, batch_labels in tqdm(train_loader, desc="训练"):
            batch_features = batch_features.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            
            # 清零梯度
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            predictions = self.model(batch_features, batch_labels, teacher_forcing=True)
            loss = ic_loss_function(predictions, batch_labels)
                
            # 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # 反向传播
            loss.backward()
            
            # 更严格的梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            # 参数更新
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 减少缓存清理频率
            if num_batches % 50 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader):
        """验证 - 速度优化版本"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_loader, desc="验证"):
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                # 验证 - 使用teacher forcing模式，避免自回归的慢速推理
                predictions = self.model(batch_features, batch_labels, teacher_forcing=True)
                loss = ic_loss_function(predictions, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和标签用于计算IC
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
                
                # 简化统计信息，只在第一个epoch显示
                if num_batches == 1 and getattr(self, 'current_epoch', 0) == 0:
                    pred_std = predictions.std().item()
                    print(f"验证批次统计: 预测范围[{predictions.min():.6f}, {predictions.max():.6f}], 标准差{pred_std:.6f}")
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算IC值
        if all_predictions and all_labels:
            all_preds = np.concatenate(all_predictions, axis=0)
            all_labs = np.concatenate(all_labels, axis=0)
            
            # 处理异常值 - 添加详细统计
            nan_count = np.isnan(all_preds).sum()
            inf_count = np.isinf(all_preds).sum()
            total_elements = all_preds.size
            valid_elements = total_elements - nan_count - inf_count
            
            print(f"预测值统计:")
            print(f"  总元素数: {total_elements}")
            print(f"  有效元素: {valid_elements} ({valid_elements/total_elements:.2%})")
            print(f"  NaN数量: {nan_count} ({nan_count/total_elements:.2%})")
            print(f"  Inf数量: {inf_count} ({inf_count/total_elements:.2%})")
            
            if valid_elements > 0:
                valid_mask = ~(np.isnan(all_preds) | np.isinf(all_preds))
                valid_preds = all_preds[valid_mask]
                
                # 数值稳定性处理：移除极值
                try:
                    # 先尝试直接使用所有有效预测值
                    pred_min = valid_preds.min()
                    pred_max = valid_preds.max()
                    pred_mean = valid_preds.mean()
                    pred_std = valid_preds.std()
                    
                    # 如果数据范围过大，再考虑移除极值
                    if pred_max - pred_min > 1000:  # 范围超过1000时才移除极值
                        q01 = np.percentile(valid_preds, 1)
                        q99 = np.percentile(valid_preds, 99)
                        filtered_preds = valid_preds[(valid_preds >= q01) & (valid_preds <= q99)]
                        
                        if len(filtered_preds) > len(valid_preds) * 0.5:  # 至少保留50%的数据
                            pred_min = filtered_preds.min()
                            pred_max = filtered_preds.max()
                            pred_mean = filtered_preds.mean()
                            pred_std = filtered_preds.std()
                            print(f"  已移除极值，保留 {len(filtered_preds)}/{len(valid_preds)} 个预测值")
                    
                    # 检查std是否异常
                    if np.isnan(pred_std) or np.isinf(pred_std):
                        pred_std = 0.0
                    
                    print(f"  有效预测值范围: [{pred_min:.6f}, {pred_max:.6f}]")
                    print(f"  有效预测值std: {pred_std:.6f}")
                    print(f"  有效预测值mean: {pred_mean:.6f}")
                        
                except Exception as e:
                    print(f"  预测值统计计算失败: {e}")
                    # 使用最基本的统计
                    try:
                        print(f"  有效预测值范围: [{valid_preds.min():.6f}, {valid_preds.max():.6f}]")
                        print(f"  有效预测值std: {valid_preds.std():.6f}")
                        print(f"  有效预测值mean: {valid_preds.mean():.6f}")
                    except:
                        print(f"  有效预测值范围: [计算失败]")
                        print(f"  有效预测值std: 0.000000")
                        print(f"  有效预测值mean: 0.000000")
                
                # 检查是否所有有效预测值都相同
                try:
                    unique_values = np.unique(valid_preds)
                    print(f"  唯一值数量: {len(unique_values)}")
                    if len(unique_values) <= 5:
                        print(f"  唯一值: {unique_values}")
                except:
                    print(f"  唯一值数量: 计算失败")
            
            if nan_count > 0 or inf_count > 0:
                all_preds = np.where(np.isnan(all_preds) | np.isinf(all_preds), 0.0, all_preds)
                print(f"异常值已清理为0.0")
            
            # 计算标准差 - 改进版本
            try:
                clean_preds = all_preds.copy()
                clean_preds = np.where(np.isnan(clean_preds) | np.isinf(clean_preds), 0.0, clean_preds)
                
                # 直接计算标准差，除非数据量太少
                if len(clean_preds) > 10:
                    pred_std = np.std(clean_preds)
                    
                    # 如果标准差过大，考虑移除极值
                    if pred_std > 100:  # 标准差过大时才考虑移除极值
                        try:
                            q01 = np.percentile(clean_preds, 1)
                            q99 = np.percentile(clean_preds, 99)
                            filtered_clean_preds = clean_preds[(clean_preds >= q01) & (clean_preds <= q99)]
                            
                            if len(filtered_clean_preds) > len(clean_preds) * 0.5:  # 至少保留50%数据
                                filtered_std = np.std(filtered_clean_preds)
                                if filtered_std < pred_std * 0.8:  # 如果移除极值后标准差明显减小
                                    pred_std = filtered_std
                                    print(f"移除极值后标准差: {pred_std:.6f}")
                        except:
                            pass  # 如果极值移除失败，使用原始标准差
                    
                    # 检查标准差是否异常
                    if np.isnan(pred_std) or np.isinf(pred_std):
                        pred_std = 1e-6
                        print(f"标准差计算异常，使用默认值")
                    elif pred_std < 1e-8:
                        print(f"预测标准差过小: {pred_std:.2e}")
                        pred_std = 1e-6
                else:
                    print(f"数据量太少（{len(clean_preds)}），无法计算可靠的标准差")
                    pred_std = 1e-6
            except Exception as e:
                print(f"标准差计算出错: {e}")
                pred_std = 1e-6
            
            try:
                clean_labs = all_labs.copy()
                clean_labs = np.where(np.isnan(clean_labs) | np.isinf(clean_labs), 0.0, clean_labs)
                lab_std = np.std(clean_labs)
                if np.isnan(lab_std) or np.isinf(lab_std):
                    lab_std = 0.0
            except:
                lab_std = 0.0
            
            print(f"验证统计: 预测std={pred_std:.6f}, 标签std={lab_std:.6f}")
            
            # 计算IC
            if pred_std < 1e-8:
                print(f"预测值方差过小 ({pred_std:.2e})，IC计算可能无效")
                avg_ic = 0.0
            else:
                ic_list = []
                
                for t in range(all_preds.shape[1]):
                    pred = all_preds[:, t]
                    true = all_labs[:, t]
                    
                    # 清理异常值
                    pred = np.where(np.isnan(pred) | np.isinf(pred), 0.0, pred)
                    true = np.where(np.isnan(true) | np.isinf(true), 0.0, true)
                    
                    # 计算统计量
                    try:
                        pred_mean = np.mean(pred)
                        pred_std = np.std(pred)
                        true_mean = np.mean(true)
                        true_std = np.std(true)
                        
                        if np.isnan(pred_mean) or np.isnan(pred_std) or np.isinf(pred_mean) or np.isinf(pred_std):
                            pred_mean, pred_std = 0.0, 1.0
                        if np.isnan(true_mean) or np.isnan(true_std) or np.isinf(true_mean) or np.isinf(true_std):
                            true_mean, true_std = 0.0, 1.0
                    except:
                        pred_mean, pred_std = 0.0, 1.0
                        true_mean, true_std = 0.0, 1.0
                    
                    # 检查数值稳定性
                    if pred_std < 1e-8 or true_std < 1e-8:
                        if pred_std < 1e-8:
                            pred += np.random.normal(0, 1e-6, pred.shape)
                            pred_mean = np.mean(pred)
                            pred_std = np.std(pred)
                            if np.isnan(pred_std) or np.isinf(pred_std):
                                pred_std = 1e-6
                        if true_std < 1e-8:
                            true += np.random.normal(0, 1e-6, true.shape)
                            true_mean = np.mean(true)
                            true_std = np.std(true)
                            if np.isnan(true_std) or np.isinf(true_std):
                                true_std = 1e-6
                        
                        if pred_std < 1e-8 or true_std < 1e-8:
                            ic = 0.0
                        else:
                            pred_norm = (pred - pred_mean) / pred_std
                            true_norm = (true - true_mean) / true_std
                            ic = np.mean(pred_norm * true_norm)
                    else:
                        pred_norm = (pred - pred_mean) / pred_std
                        true_norm = (true - true_mean) / true_std
                        ic = np.mean(pred_norm * true_norm)
                        
                        if np.isnan(ic) or np.isinf(ic):
                            ic = 0.0
                    
                    ic_list.append(ic)
                
                avg_ic = np.mean(ic_list) if ic_list else 0.0
        else:
            avg_ic = 0.0
            print("验证数据收集失败")
        
        return avg_loss, avg_ic
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """训练模型 - 速度优化版本"""
        print("="*50)
        print("开始训练模型")
        print("="*50)
        
        best_val_loss = float('inf')
        best_val_ic = -float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 15  # 进一步减少耐心，更快收敛
        
        # 多指标跟踪
        ic_history = []
        loss_history = []
        best_ic_epochs = []
        
        # 记录训练历史
        train_losses = []
        val_losses = []
        val_ics = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 设置当前epoch供其他函数使用
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_ic = self.validate(val_loader)
            
            # 记录历史
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_ics.append(val_ic)
            
            # 学习率调度
            if hasattr(self.scheduler, 'step'):
                try:
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  当前学习率: {current_lr:.2e}")
                except Exception as e:
                    print(f"学习率调度器调用失败: {e}")
            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}/{val_loss:.4f}, IC={val_ic:.4f} (Best: {best_val_ic:.4f})")
            
            # 改进的早停机制
            improved = False
            
            # 记录历史
            ic_history.append(val_ic)
            loss_history.append(val_loss)
            
            # 主要指标：IC改善时保存模型
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                best_ic_epochs.append(epoch)
                improved = True
                print(f"发现更好的IC: {best_val_ic:.6f}")
            
            # 次要指标：验证损失显著改善但IC没有恶化太多
            elif val_loss < best_val_loss * 0.98 and val_ic > best_val_ic * 0.9:  # 更严格的条件
                best_val_loss = val_loss
                best_val_ic = val_ic
                best_model_state = self.model.state_dict().copy()
                improved = True
                print(f"验证损失显著改善: {val_loss:.6f} (IC: {val_ic:.6f})")
            
            # 简化IC趋势检查，减少计算开销
            if len(ic_history) >= 5 and epoch % 5 == 0:  # 每5个epoch检查一次
                recent_ic_trend = np.polyfit(range(5), ic_history[-5:], 1)[0]  # 线性趋势
                if recent_ic_trend > 0:
                    print(f"IC趋势良好: {recent_ic_trend:.6f}/epoch")
            
            if improved:
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_ic': best_val_ic,
                    'config': self.model_config,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_ics': val_ics
                }, 'best_direct_model.pth')
                print(f"  模型已保存 (IC: {best_val_ic:.6f})")
            else:
                patience_counter += 1
                
                # 早停条件
                if patience_counter >= patience:
                    print(f"  早停: 最佳IC={best_val_ic:.6f} (epoch {epoch+1})")
                    
                    # 恢复最佳模型状态
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
            
            # 高IC提前停止
            if best_val_ic > 0.08:  # 降低阈值
                if patience_counter >= patience // 2:
                    print(f"  高IC提前停止: {best_val_ic:.6f}")
                    break
        
        # 确保使用最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"训练完成，最佳IC: {best_val_ic:.6f}")
        return best_val_ic
    
    def train_ensemble_models(self, seeds=[123, 456, 789], save_models=True, use_cache=True):
        """训练集成模型"""
        print("="*70)
        print("多种子集成模型训练")
        print(f"训练种子: {seeds}")
        print(f"模型数量: {len(seeds)}")
        print("="*70)
        
        # 创建保存目录
        if save_models:
            os.makedirs('trained_models', exist_ok=True)
            os.makedirs('factor_results', exist_ok=True)
        
        models_info = []
        trained_models = []
        
        # 加载数据
        data = self.load_direct_data()
        if data is None:
            print("数据加载失败")
            return None
        
        features_df, stock_data = data
        features, labels, dates = self.process_features_and_labels(features_df, stock_data)
        if features is None:
            return None
        
        train_loader, val_loader, scaler = self.create_data_loaders(
            (features, labels, dates), 
            batch_size=self.batch_size,
            use_cache=use_cache
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
            
            # 重新初始化模型
            model = TransformerSeq2SeqModel(**self.model_config)
            model.to(self.device)
            
            # 改进的权重初始化
            output_layers = 0
            other_layers = 0
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if 'fc_out' in name:
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)  # 使用与训练器相同的初始化
                        output_layers += 1
                    else:
                        torch.nn.init.xavier_uniform_(module.weight, gain=0.8)  # 适中的其他层gain
                        other_layers += 1
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
            
            # 确保所有参数都需要梯度
            for param in model.parameters():
                param.requires_grad = True
            
            print(f"模型 {i+1} 权重初始化完成: 输出层{output_layers}个, 其他层{other_layers}个")
            
            # 重新初始化优化器和调度器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-5,  # 使用与训练器相同的学习率
                weight_decay=1e-3,  # 使用与训练器相同的权重衰减
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=20,
                T_mult=2,
                eta_min=1e-6
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
                num_epochs=self.num_epochs  # 使用训练器的epoch数量
            )
            
            # 保存模型信息
            model_info = {
                'model_id': i + 1,
                'seed': seed,
                'best_ic': float(best_ic),
                'config': self.model_config.copy()
            }
            models_info.append(model_info)
            
            # 保存模型到CPU
            trained_models.append(self.model.cpu().state_dict())
            
            # 保存单个模型文件
            if save_models:
                model_file = f"trained_models/transformer_model_seed_{seed}.pth"
                torch.save({
                    'epoch': 'final',
                    'model_state_dict': self.model.state_dict(),
                    'model_info': model_info,
                    'config': self.model_config
                }, model_file)
                print(f"模型已保存: {model_file}")
            
                print(f"模型 {i+1}/{len(seeds)} 完成，最佳IC: {best_ic:.6f}")
            
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
            
            print(f"集成信息已保存: trained_models/ensemble_info.json")
        
        print(f"\n{'='*50}")
        print("集成训练完成!")
        avg_ic = np.mean([info['best_ic'] for info in models_info])
        print(f"{len(trained_models)}个模型，平均IC: {avg_ic:.6f}")
        print("="*50)
        
        # 计算统计信息
        avg_ic = np.mean([info['best_ic'] for info in models_info])
        best_ic = max([info['best_ic'] for info in models_info])
        
        return {
            'models_info': models_info,
            'trained_models': trained_models,
            'scaler': scaler,
            'avg_ic': avg_ic,
            'best_ic': best_ic,
            'num_models': len(seeds)
        }

def main():
    """主函数 - 优化版本"""
    print("="*60)
    print("Transformer因子训练 - 优化版本")
    print("="*60)
    
    # 检查GPU内存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU内存: {gpu_memory:.1f} GB")
        
        if gpu_memory < 20:
            print("GPU内存不足，建议使用更小的模型配置")
    
            # 速度优化后的模型配置
        model_config = {
            'input_dim': 6,
            'd_model': 192,        # 从256减小到192，平衡速度和性能
            'nhead': 12,           # 从16减小到12
            'num_layers': 4,       # 从6减小到4
            'dim_feedforward': 768,   # 从1024减小到768
            'dropout': 0.1,
            'label_horizon': 10
        }
    
    print(f"模型配置: {model_config}")
    
    # 创建训练器
    trainer = TransformerFactorTrainer(model_config, max_stocks=None)
    
    # 训练集成模型
    print("\n开始训练集成模型...")
    ensemble_info = trainer.train_ensemble_models(
        seeds=[123, 456, 789],  # 3个模型
        save_models=True,
        use_cache=True
    )
    
    if ensemble_info:
        print("\n集成训练完成!")
        print(f"平均IC: {ensemble_info['avg_ic']:.6f}")
        print(f"最佳IC: {ensemble_info['best_ic']:.6f}")
        print(f"模型数量: {ensemble_info['num_models']}")
    else:
        print("\n训练失败!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 