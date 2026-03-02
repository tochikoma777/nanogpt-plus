#!/usr/bin/env python
"""
示例2: 使用LoRA微调预训练模型
"""

import torch

from nanogpt_plus.config import ModelConfig, TrainingConfig
from nanogpt_plus.data.loaders.text_loader import InstructionDataset, create_dataloader
from nanogpt_plus.models.gpt import create_model
from nanogpt_plus.models.lora import inject_lora, get_lora_params, print_trainable_parameters
from nanogpt_plus.training.trainer import Trainer


def create_sample_instruction_data():
    """创建示例指令数据（实际应从文件加载）"""
    return [
        {
            "instruction": "解释什么是机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习规律，而无需明确编程..."
        },
        {
            "instruction": "将以下英文翻译成中文",
            "input": "The quick brown fox jumps over the lazy dog.",
            "output": "敏捷的棕色狐狸跳过了懒惰的狗。"
        },
        # ... 更多数据
    ] * 100  # 重复以创建足够数据


def main():
    torch.manual_seed(42)
    
    # 加载预训练模型（或从头创建）
    model_config = ModelConfig(
        name="gpt2-medium-lora",
        vocab_size=50257,
        block_size=1024,
        n_layer=24,  # Medium
        n_head=16,
        n_embd=1024,
        dropout=0.1,
        bias=True,
    )
    
    print("创建基础模型...")
    model = create_model(model_config)
    
    # 注入LoRA
    print("注入LoRA模块...")
    inject_lora(
        model,
        target_modules=["c_attn", "c_proj", "c_fc", "c_proj"],  # 注意力层和MLP层
        r=16,              # LoRA秩
        lora_alpha=32,     # 缩放因子
        lora_dropout=0.05,
    )
    
    # 冻结非LoRA参数
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    # 打印可训练参数
    print_trainable_parameters(model)
    
    # 创建优化器（只优化LoRA参数）
    lora_params = get_lora_params(model)
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
    
    # 加载指令数据
    print("加载指令数据...")
    raw_data = create_sample_instruction_data()
    dataset = InstructionDataset(
        data=raw_data,
        max_length=512,  # 指令通常较短
        mask_inputs=True,
    )
    train_loader = create_dataloader(dataset, batch_size=4, shuffle=True)
    
    # 训练配置
    training_config = TrainingConfig(
        max_iters=2000,
        batch_size=4,
        block_size=512,
        learning_rate=1e-4,
        warmup_iters=100,
        eval_interval=500,
        out_dir="outputs/lora_experiment",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        optimizer=optimizer,  # 传入自定义优化器
    )
    
    print("开始LoRA微调...")
    trainer.train()
    
    # 保存LoRA权重（可合并到基础模型）
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, f"{training_config.out_dir}/lora_weights.pt")
    print(f"LoRA权重已保存！")


if __name__ == "__main__":
    main()
