#!/usr/bin/env python
"""
示例1: 从头训练GPT模型
"""

import torch

from nanogpt_plus.config import ModelConfig, TrainingConfig
from nanogpt_plus.data.loaders.text_loader import get_openwebtext_dataloader
from nanogpt_plus.models.gpt import create_model
from nanogpt_plus.training.trainer import Trainer


def main():
    # 设置随机种子
    torch.manual_seed(1337)
    
    # 模型配置（GPT-2 Small）
    model_config = ModelConfig(
        name="gpt2-small-custom",
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=False,
    )
    
    # 训练配置
    training_config = TrainingConfig(
        max_iters=10000,  # 演示用，实际可设为100000+
        batch_size=8,     # 根据显存调整
        block_size=1024,
        learning_rate=6e-4,
        warmup_iters=100,
        eval_interval=500,
        checkpoint_interval=1000,
        out_dir="outputs/experiment_01",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    
    print(f"创建模型: {model_config.name}")
    print(f"参数量: ~{sum(p.numel() for p in create_model(model_config).parameters())/1e6:.1f}M")
    
    # 创建模型
    model = create_model(model_config)
    
    # 加载数据
    print("加载数据...")
    train_loader = get_openwebtext_dataloader(
        batch_size=training_config.batch_size,
        block_size=training_config.block_size,
    )
    
    # 创建验证集（简化：用训练集的一部分）
    val_loader = get_openwebtext_dataloader(
        batch_size=training_config.batch_size,
        block_size=training_config.block_size,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    print(f"训练完成！模型保存在: {training_config.out_dir}")


if __name__ == "__main__":
    main()
