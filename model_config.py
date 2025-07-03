#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer因子模型配置文件
"""

# 基础模型配置
BASE_CONFIG = {
    'model_params': {
        'input_dim': 6,         # 6大基本特征：OHLC + VWAP + Amount
        'd_model': 128,         # Transformer模型维度
        'nhead': 8,             # 多头注意力头数
        'num_layers': 4,        # Encoder层数
        'dim_feedforward': 512, # 前馈网络维度
        'dropout': 0.1          # Dropout率
    },
    'seq_len': 40,              # 输入序列长度（过去40个交易日）
    'batch_size': 64,           # 训练批大小
    'learning_rate': 1e-4,      # 初始学习率
    'weight_decay': 1e-5,       # 权重衰减
    'max_epochs': 100,          # 最大训练轮数
    'patience': 15,             # 早停耐心值
    'grad_clip': 1.0           # 梯度裁剪阈值
}

# 小规模测试配置
SMALL_CONFIG = {
    'model_params': {
        'input_dim': 6,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1
    },
    'seq_len': 20,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 50,
    'patience': 10,
    'grad_clip': 1.0
}

# 大规模配置
LARGE_CONFIG = {
    'model_params': {
        'input_dim': 6,
        'd_model': 256,
        'nhead': 16,
        'num_layers': 6,
        'dim_feedforward': 1024,
        'dropout': 0.1
    },
    'seq_len': 60,
    'batch_size': 32,
    'learning_rate': 5e-5,
    'weight_decay': 1e-5,
    'max_epochs': 200,
    'patience': 20,
    'grad_clip': 1.0
}

# 数据配置
DATA_CONFIG = {
    'equal_volume_file': 'equal_volume_features_all.csv',
    'stock_data_file': 'stock_price_vol_d.txt',
    'feature_columns': ['open', 'high', 'low', 'close', 'vwap', 'amount'],
    'target_days': 10,          # 预测未来10日收益
    'min_samples_per_stock': 100, # 每只股票最少样本数
    'train_ratio': 0.8,         # 训练集比例
    'sample_interval': 5        # 每5个交易日采样一次
} 