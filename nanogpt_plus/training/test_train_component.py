# 检查所有文件
print("=== 检查训练文件 ===")
print("=== 测试导入 ===")
from nanogpt_plus import Trainer, CosineWarmupScheduler
from nanogpt_plus import save_checkpoint, load_checkpoint, get_logger
print('✅ 所有训练组件导入成功')


print("=== 快速训练测试 ===")
import torch
import tempfile
from nanogpt_plus import create_model, ModelConfig, TrainingConfig, Trainer
from nanogpt_plus.data import TextDataset, create_dataloader

# 创建测试数据
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write('Test data for GPT training. ' * 200)
    path = f.name

# 创建小模型
model = create_model(ModelConfig(n_layer=2, n_embd=64, n_head=4, vocab_size=50257))

# 创建数据
dataset = TextDataset(path, is_file=True, block_size=16, stride=8)
loader = create_dataloader(dataset, batch_size=2, shuffle=True)

# 训练配置（极简）
cfg = TrainingConfig(
    max_iters=10,
    batch_size=2,
    block_size=16,
    learning_rate=1e-3,
    warmup_iters=2,
    log_interval=5,
    checkpoint_interval=10,
    out_dir='outputs/test',
    device='cpu',
    dtype='float32',
    compile=False,
)

# 训练
trainer = Trainer(model, cfg, loader)
trainer.train()

print('\n🎉 训练系统测试通过！')