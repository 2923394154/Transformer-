#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
损失函数测试脚本 - 针对IC为负的情况测试不同解决方案
"""

from transformer_factor_model import FactorTrainer
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_loss_functions():
    """测试不同损失函数的效果"""
    
    print("损失函数效果测试")
    print("=" * 60)
    
    # 不同损失函数配置
    loss_types = ['ic', 'rank_ic', 'mse', 'huber', 'combined', 'directional']
    
    results = {}
    
    for loss_type in loss_types:
        print(f"\n测试损失函数: {loss_type}")
        print("-" * 40)
        
        # 配置
        config = {
            'model_params': {
                'input_dim': 6,
                'd_model': 64,      # 小模型快速测试
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 128,
                'dropout': 0.1
            },
            'seq_len': 20,
            'batch_size': 128,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'use_amp': torch.cuda.is_available(),
            'loss_type': loss_type
        }
        
        try:
            # 创建训练器
            trainer = FactorTrainer(config)
            
            # 准备数据
            train_loader, val_loader = trainer.prepare_data()
            
            if train_loader is None:
                print(f"数据准备失败，跳过 {loss_type}")
                continue
            
            # 短期训练测试（10个epochs）
            print(f"开始训练 ({loss_type} 损失函数)...")
            best_ic = trainer.train(train_loader, val_loader, num_epochs=10)
            
            # 记录结果
            results[loss_type] = {
                'best_ic': best_ic,
                'final_ic': trainer.val_ics[-1] if trainer.val_ics else 0,
                'ic_trend': trainer.val_ics[-5:] if len(trainer.val_ics) >= 5 else trainer.val_ics,
                'convergence_speed': len([ic for ic in trainer.val_ics if ic > 0.01])  # 正IC的轮次数
            }
            
            print(f"{loss_type} 完成 - 最佳IC: {best_ic:.6f}")
            
        except Exception as e:
            print(f"{loss_type} 测试失败: {e}")
            results[loss_type] = {'error': str(e)}
    
    # 分析结果
    print("\n" + "=" * 60)
    print("测试结果分析")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        # 找出表现最好的损失函数
        best_loss_type = max(valid_results.keys(), 
                           key=lambda x: valid_results[x]['best_ic'])
        
        print(f"\n推荐损失函数: {best_loss_type}")
        print(f"最佳IC: {valid_results[best_loss_type]['best_ic']:.6f}")
        
        print("\n各损失函数详细结果:")
        for loss_type, result in valid_results.items():
            print(f"\n{loss_type}:")
            print(f"  最佳IC: {result['best_ic']:.6f}")
            print(f"  最终IC: {result['final_ic']:.6f}")
            print(f"  收敛速度: {result['convergence_speed']} epochs")
            print(f"  近期IC趋势: {[f'{ic:.4f}' for ic in result['ic_trend']]}")
        
        # 绘制对比图
        plot_loss_comparison(valid_results)
        
        return best_loss_type
    
    else:
        print("所有损失函数测试都失败了，请检查数据和配置")
        return None

def plot_loss_comparison(results):
    """绘制损失函数对比图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 最佳IC对比
    loss_types = list(results.keys())
    best_ics = [results[lt]['best_ic'] for lt in loss_types]
    final_ics = [results[lt]['final_ic'] for lt in loss_types]
    
    x = np.arange(len(loss_types))
    width = 0.35
    
    ax1.bar(x - width/2, best_ics, width, label='最佳IC', alpha=0.7)
    ax1.bar(x + width/2, final_ics, width, label='最终IC', alpha=0.7)
    ax1.set_xlabel('损失函数类型')
    ax1.set_ylabel('IC值')
    ax1.set_title('不同损失函数的IC表现')
    ax1.set_xticks(x)
    ax1.set_xticklabels(loss_types, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 收敛速度对比
    convergence_speeds = [results[lt]['convergence_speed'] for lt in loss_types]
    
    ax2.bar(loss_types, convergence_speeds, alpha=0.7, color='green')
    ax2.set_xlabel('损失函数类型')
    ax2.set_ylabel('正IC轮次数')
    ax2.set_title('收敛速度对比（正IC的轮次数）')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_function_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n对比图已保存为 loss_function_comparison.png")

def quick_ic_diagnosis():
    """快速IC问题诊断"""
    
    print("IC问题快速诊断")
    print("=" * 40)
    
    # 使用简单配置快速测试
    config = {
        'model_params': {
            'input_dim': 6,
            'd_model': 32,
            'nhead': 2,
            'num_layers': 1,
            'dim_feedforward': 64,
            'dropout': 0.1
        },
        'seq_len': 10,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'use_amp': False,
        'loss_type': 'mse'  # 先用MSE检查基础预测能力
    }
    
    trainer = FactorTrainer(config)
    train_loader, val_loader = trainer.prepare_data()
    
    if train_loader is None:
        print("数据加载失败")
        return
    
    # 快速训练5轮
    trainer.train(train_loader, val_loader, num_epochs=5)
    
    if trainer.val_ics:
        avg_ic = sum(trainer.val_ics) / len(trainer.val_ics)
        
        print(f"\n诊断结果:")
        print(f"平均IC: {avg_ic:.6f}")
        
        if avg_ic < -0.05:
            print("⚠️  IC严重为负，建议检查：")
            print("   1. 收益率标签计算是否正确")
            print("   2. 特征与标签的时间对齐")
            print("   3. 数据是否存在look-ahead bias")
            print("   4. 特征工程是否合理")
        elif avg_ic < 0:
            print("⚠️  IC轻微为负，建议：")
            print("   1. 使用组合损失函数")
            print("   2. 调整模型复杂度")
            print("   3. 增加正则化")
        else:
            print("✅ IC为正，模型基础方向正确")
            print("   建议使用IC或RankIC损失函数")

def main():
    """主函数"""
    
    print("IC负值问题解决方案测试")
    print("=" * 60)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--diagnosis":
            # 快速诊断模式
            quick_ic_diagnosis()
        elif sys.argv[1] == "--test":
            # 完整测试模式
            best_loss = test_loss_functions()
            if best_loss:
                print(f"\n✅ 推荐在实际训练中使用: {best_loss} 损失函数")
        else:
            print("用法:")
            print("  python loss_function_test.py --diagnosis  # 快速诊断")
            print("  python loss_function_test.py --test       # 完整测试")
    else:
        # 默认运行诊断
        quick_ic_diagnosis()

if __name__ == "__main__":
    main() 