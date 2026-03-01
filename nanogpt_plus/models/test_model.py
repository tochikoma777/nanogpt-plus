from nanogpt_plus.models import GPT, create_model, LayerNorm
from nanogpt_plus.config import ModelConfig
import torch

# 创建一个小模型测试
cfg = ModelConfig(n_layer=2, n_embd=64, n_head=4, vocab_size=100)
model = create_model(cfg)

print(f'✓ 模型创建成功')
print(f'  类型: {type(model).__name__}')
print(f'  参数量: {model.get_num_params()/1e6:.2f}M')

# 测试前向传播
x = torch.randint(0, 100, (2, 10))  # batch=2, seq_len=10
logits, loss = model(x)
print(f'✓ 前向传播成功')
print(f'  输入形状: {x.shape}')
print(f'  输出形状: {logits.shape}')
print(f'  损失: {loss} (None，因为没有提供targets)')

# 测试生成
model.eval()
generated = model.generate(x[:, :1], max_new_tokens=5)
print(f'✓ 生成成功')
print(f'  生成序列长度: {generated.shape[1]}')