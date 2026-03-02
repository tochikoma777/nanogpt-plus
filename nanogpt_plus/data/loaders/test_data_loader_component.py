print("=== 检查数据文件 ===")
print("=== 测试完整导入 ===")
from nanogpt_plus.data import (
    TextDataset, 
    InstructionDataset, 
    create_dataloader,
    instruction_collate_fn,
)
print('✅ 数据模块导入成功')


print("=== 运行完整数据测试 ===")
import tempfile, json
from nanogpt_plus.data import TextDataset, InstructionDataset, create_dataloader

# 快速测试
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write('Test data for GPT training. ' * 50)
    text_path = f.name

dataset = TextDataset(text_path, is_file=True, block_size=32)
loader = create_dataloader(dataset, batch_size=2)
batch = next(iter(loader))

print(f'✅ 文本数据: 批次形状 {batch["input_ids"].shape}')

# 指令数据
instructions = [{'instruction': 'Hi', 'input': '', 'output': 'Hello'}] * 5
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
    json.dump(instructions, f)
    json_path = f.name

inst_dataset = InstructionDataset(json_path, max_length=32)
inst_loader = create_dataloader(inst_dataset, batch_size=2)
inst_batch = next(iter(inst_loader))

print(f'✅ 指令数据: 批次形状 {inst_batch["input_ids"].shape}')
print('\n🎉 数据系统全部正常！')