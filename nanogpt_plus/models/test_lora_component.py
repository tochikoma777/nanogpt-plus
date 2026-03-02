print("=== 检查LoRA文件 ===")
print("=== 测试完整导入 ===")

from nanogpt_plus.models import (
    GPT, create_model, 
    LoRALinear, inject_lora, setup_lora_for_training,
    print_trainable_parameters
)
print('✅ 所有LoRA组件导入成功')


print("=== 运行完整LoRA测试 ===")

import torch
from nanogpt_plus.models import create_model, setup_lora_for_training
from nanogpt_plus.config import ModelConfig

# 创建并配置LoRA
cfg = ModelConfig(n_layer=2, n_embd=256, n_head=4)
model = create_model(cfg)

print('原始模型参数量:', model.get_num_params()/1e6, 'M')

lora_params = setup_lora_for_training(model, r=8, lora_alpha=16)

# 测试训练步骤
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
x = torch.randint(0, cfg.vocab_size, (2, 10))
logits, loss = model(x, x)  # 自回归：输入=目标

loss.backward()
optimizer.step()

print('\n✅ LoRA训练步骤成功！')
print(f'损失: {loss.item():.4f}')