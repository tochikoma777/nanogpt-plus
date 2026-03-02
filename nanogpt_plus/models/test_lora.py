# 测试LoRA功能

import torch
from nanogpt_plus.models import LoRALinear, inject_lora, print_trainable_parameters, setup_lora_for_training
from nanogpt_plus.models import create_model
from nanogpt_plus.config import ModelConfig

print('=== 测试1: 单独LoRA层 ===')
# 创建标准线性层和LoRA层对比
linear = torch.nn.Linear(768, 768)
lora = LoRALinear(768, 768, r=8, lora_alpha=16)

print(f'标准Linear参数量: {sum(p.numel() for p in linear.parameters()):,}')
print(f'LoRA总参数量: {sum(p.numel() for p in lora.parameters()):,}')
print(f'LoRA可训练参数量: {sum(p.numel() for p in lora.parameters() if p.requires_grad):,}')

# 前向传播测试
x = torch.randn(2, 10, 768)
out = lora(x)
print(f'输入形状: {x.shape}, 输出形状: {out.shape}')

print('\n=== 测试2: 注入LoRA到GPT模型 ===')
# 创建小模型
cfg = ModelConfig(n_layer=2, n_embd=128, n_head=4, vocab_size=1000)
model = create_model(cfg)

print('注入前:')
print_trainable_parameters(model)

# 一键设置LoRA
lora_params = setup_lora_for_training(model, r=4, lora_alpha=8)

print(f'\n返回的LoRA参数数量: {len(lora_params)}')
print('✅ LoRA模块工作正常！')