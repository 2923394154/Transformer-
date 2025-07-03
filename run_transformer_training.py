#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行Transformer因子训练的主脚本
"""

from transformer_factor_model import FactorTrainer
import torch

def main():
    """运行Transformer因子训练"""
    
    print("Transformer因子合成训练系统")
    print("=" * 60)
    
    # 检查是否有CUDA设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置 - 根据您的需求调整
    config = {
        'model_params': {
            'input_dim': 6,         # 6大基本特征
            'd_model': 128,         # 模型维度
            'nhead': 8,             # 注意力头数  
            'num_layers': 4,        # Encoder层数
            'dim_feedforward': 512, # 前馈网络维度
            'dropout': 0.1          # Dropout率
        },
        'seq_len': 40,              # 过去40个交易日
        'batch_size': 64,           # 批大小
        'learning_rate': 1e-4,      # 学习率
        'weight_decay': 1e-5        # 权重衰减
    }
    
    print("\n模型配置参数:")
    print("-" * 40)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # 创建训练器
    try:
        trainer = FactorTrainer(config)
        print(f"\n模型参数总数: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        # 准备数据
        print("\n正在准备训练数据...")
        train_loader, val_loader = trainer.prepare_data()
        
        if train_loader is None or val_loader is None:
            print("\n数据准备失败！")
            print("请确保以下文件存在:")
            print("1. equal_volume_features_all.csv (运行 equal_volume_kline.py 生成)")
            print("2. stock_price_vol_d.txt (原始股票数据)")
            return
        
        print(f"数据准备完成！")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        
        # 开始训练
        print("\n开始模型训练...")
        print("=" * 60)
        
        best_ic = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader, 
            num_epochs=100
        )
        
        print("\n训练完成！")
        print("=" * 60)
        print(f"最佳验证IC: {best_ic:.6f}")
        print("模型文件已保存: best_transformer_model.pth")
        print("训练曲线已保存: training_curves.png")
        
        return trainer
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_test():
    """快速测试模式 - 使用较小的模型和数据"""
    
    print("快速测试模式")
    print("=" * 40)
    
    # 小规模配置用于快速测试
    test_config = {
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
        'weight_decay': 1e-5
    }
    
    trainer = FactorTrainer(test_config)
    
    # 准备数据
    train_loader, val_loader = trainer.prepare_data()
    
    if train_loader is None:
        print("数据准备失败，无法进行测试")
        return None
    
    # 运行少量epochs进行测试
    print("运行快速测试（5个epochs）...")
    trainer.train(train_loader, val_loader, num_epochs=5)
    
    return trainer

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 快速测试模式
        trainer = quick_test()
    else:
        # 完整训练模式
        trainer = main()
    
    if trainer is not None:
        print("\n脚本执行完成！")
    else:
        print("\n脚本执行失败！") 