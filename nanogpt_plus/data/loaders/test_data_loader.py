# 测试数据加载
import torch
from nanogpt_plus.data import TextDataset, InstructionDataset, create_dataloader

print('=== 测试1: 通用文本数据集 ===')
# 创建临时文本
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write('This is a test sentence. ' * 100)
    temp_path = f.name

dataset = TextDataset(
    data=temp_path,
    is_file=True,
    block_size=16,
    stride=8,
)
print(f'样本数: {len(dataset)}')

# 取一个样本
sample = dataset[0]
print(f'输入形状: {sample["input_ids"].shape}')
print(f'目标形状: {sample["labels"].shape}')

# 创建dataloader
loader = create_dataloader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))
print(f'批次输入形状: {batch["input_ids"].shape}')

print('\n=== 测试2: 指令数据集 ===')
# 创建示例指令数据
import json
instructions = [
    {'instruction': '解释AI', 'input': '', 'output': 'AI是人工智能...'},
    {'instruction': '翻译', 'input': 'Hello', 'output': '你好'},
    {'instruction': '写代码', 'input': 'Python hello world', 'output': 'print(\"Hello\")'},
] * 10  # 30条

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
    json.dump(instructions, f)
    json_path = f.name

inst_dataset = InstructionDataset(
    data=json_path,
    max_length=64,
    mask_inputs=True,
)
print(f'指令样本数: {len(inst_dataset)}')

# 检查mask
sample = inst_dataset[0]
print(f'输入token数: {len(sample["input_ids"])}')
print(f'Reply mask为1的数量: {sample["loss_mask"].sum().item()}')  # 应该只有回复部分是1

# 创建dataloader
from nanogpt_plus.data import instruction_collate_fn
inst_loader = create_dataloader(inst_dataset, batch_size=4, collate_fn=instruction_collate_fn)
inst_batch = next(iter(inst_loader))
print(f'指令批次输入: {inst_batch["input_ids"].shape}')
print(f'指令批次标签: {inst_batch["labels"].shape}')

print('\n✅ 数据加载系统测试通过！')