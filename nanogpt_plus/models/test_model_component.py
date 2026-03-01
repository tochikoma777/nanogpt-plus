#完成检查清单

from nanogpt_plus.models import GPT, create_model, LayerNorm, CausalSelfAttention, Block
from nanogpt_plus.config import ModelConfig
import torch

print("=== 检查模型文件 ===")
# 创建配置
cfg = ModelConfig(n_layer=2, n_embd=128, n_head=4, block_size=64, vocab_size=1000)

# 测试各组件
print('测试LayerNorm...')
ln = LayerNorm(128, bias=True)
x = torch.randn(2, 10, 128)
y = ln(x)
assert y.shape == x.shape
print('✓ LayerNorm正常')

print('测试CausalSelfAttention...')
attn = CausalSelfAttention(cfg)
x = torch.randn(2, 10, 128)
y = attn(x)
assert y.shape == x.shape
print('✓ Attention正常')

print('测试Block...')
block = Block(cfg)
y = block(x)
assert y.shape == x.shape
print('✓ Block正常')

print('测试完整GPT...')
model = create_model(cfg)
logits, _ = model(torch.randint(0, 1000, (2, 10)))
assert logits.shape == (2, 10, 1000)
print('✓ GPT模型正常')
print('\n🎉 所有组件测试通过！')

print("\n=== 显示模型结构 ===")
from nanogpt_plus.models import create_model
from nanogpt_plus.config import ModelConfig

cfg = ModelConfig()
model = create_model(cfg)
print(f'模型: {cfg.name}')
print(f'总参数量: {model.get_num_params()/1e6:.1f}M')
print(f'层数: {cfg.n_layer}')
print(f'注意力头数: {cfg.n_head}')
print(f'嵌入维度: {cfg.n_embd}')