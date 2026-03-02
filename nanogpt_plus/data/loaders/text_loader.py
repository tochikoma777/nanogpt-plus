"""
文本数据加载器
支持：
1. 通用文本（预训练）：滑动窗口切分
2. 指令数据（微调）：Alpaca格式，模板化
"""

import os
import json
import random
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Union

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class TextDataset(Dataset):
    """
    通用文本数据集（用于预训练）
    
    使用滑动窗口将长文本切分为固定长度序列
    
    示例：
        文本: "The cat sat on the mat and looked at the dog..."
        block_size=4, stride=2:
            样本1: [The, cat, sat, on] -> 目标: [cat, sat, on, the]
            样本2: [sat, on, the, mat] -> 目标: [on, the, mat, and]
            （重叠2个token，充分利用数据）
    """
    
    def __init__(
        self,
        data: Union[str, Path, List[int]],  # 文本路径或直接token列表
        tokenizer_name: str = "gpt2",
        block_size: int = 1024,
        stride: Optional[int] = None,  # None则等于block_size（不重叠）
        is_file: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        
        # 初始化tokenizer（BPE编码）
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.eot_token = self.tokenizer.eot_token  # <|endoftext|>的ID
        
        # 加载并编码数据
        if is_file and isinstance(data, (str, Path)) and os.path.exists(data):
            with open(data, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"加载文件: {data}, 大小: {len(text):,} 字符")
        elif isinstance(data, str):
            text = data
        else:
            text = ""
        
        # 编码为token IDs
        print("编码中...")
        self.tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        print(f"Token数量: {len(self.tokens):,}")
        
        # 计算样本数
        n_tokens = len(self.tokens)
        if n_tokens <= block_size:
            self.n_samples = 1
        else:
            self.n_samples = (n_tokens - block_size) // self.stride + 1
        
        print(f"生成样本数: {self.n_samples:,} (block_size={block_size}, stride={self.stride})")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 计算起始位置
        start = idx * self.stride
        end = start + self.block_size + 1  # 多取1个作为目标
        
        # 边界检查
        if end > len(self.tokens):
            # 不足时填充结束符
            chunk = self.tokens[start:] + [self.eot_token] * (end - len(self.tokens))
        else:
            chunk = self.tokens[start:end]
        
        # 确保长度正确
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [self.eot_token] * (self.block_size + 1 - len(chunk))
        
        # 输入是[:-1]，目标是[1:]（预测下一个token）
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return {
            "input_ids": x,
            "labels": y,
            "length": torch.tensor(len(chunk) - 1),  # 实际长度（用于mask）
        }


class InstructionDataset(Dataset):
    """
    指令微调数据集
    支持Alpaca格式：instruction, input, output
    
    关键特性：
    1. 统一模板格式化（指令+输入+回复）
    2. 支持损失屏蔽（只计算回复部分的损失）
    3. 自动截断/填充到固定长度
    
    数据格式示例：
    {
        "instruction": "解释什么是机器学习",
        "input": "",  # 可选
        "output": "机器学习是人工智能的一个分支..."
    }
    """
    
    def __init__(
        self,
        data: Union[str, Path, List[Dict]],  # JSON文件路径或数据列表
        tokenizer_name: str = "gpt2",
        max_length: int = 512,  # 指令通常比预训练短
        mask_inputs: bool = True,  # 是否屏蔽输入部分的损失
        template: str = "alpaca",  # 模板类型
    ):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_length = max_length
        self.mask_inputs = mask_inputs
        self.eot_token = self.tokenizer.eot_token
        
        # 加载数据
        if isinstance(data, (str, Path)) and os.path.exists(data):
            with open(data, 'r', encoding='utf-8') as f:
                if str(data).endswith('.jsonl'):
                    # JSONL格式：每行一个JSON
                    self.raw_data = [json.loads(line) for line in f]
                else:
                    # JSON格式：整个文件是一个列表
                    self.raw_data = json.load(f)
        elif isinstance(data, list):
            self.raw_data = data
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
        
        print(f"加载 {len(self.raw_data)} 条指令数据")
        
        # 预处理和编码
        self.samples = []
        skipped = 0
        
        for item in self.raw_data:
            # 格式化文本
            text = self._format_text(item, template)
            
            # 编码
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            
            # 过长则截断
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                skipped += 1
            
            # 找到回复开始位置（用于mask）
            response_start = self._find_response_start(tokens, item, template)
            
            self.samples.append({
                "tokens": tokens,
                "response_start": response_start,  # 损失计算的起始位置
                "length": len(tokens),
            })
        
        if skipped > 0:
            print(f"警告: {skipped} 条数据因过长被截断")
        print(f"有效样本: {len(self.samples)}")
    
    def _format_text(self, item: Dict, template: str) -> str:
        """格式化为统一模板"""
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        
        if template == "alpaca":
            # Alpaca标准模板
            if input_text:
                prompt = (
                    f"Below is an instruction that describes a task, paired with an input that provides further context. "
                    f"Write a response that appropriately completes the request.\\n\\n"
                    f"### Instruction:\\n{instruction}\\n\\n"
                    f"### Input:\\n{input_text}\\n\\n"
                    f"### Response:\\n"
                )
            else:
                prompt = (
                    f"Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request.\\n\\n"
                    f"### Instruction:\\n{instruction}\\n\\n"
                    f"### Response:\\n"
                )
            
            # 完整文本：提示 + 回复 + 结束符
            full_text = prompt + output + "<|endoftext|>"
            
        elif template == "simple":
            # 简化模板
            full_text = f"指令: {instruction}\\n输入: {input_text}\\n回复: {output}<|endoftext|>"
        
        else:
            raise ValueError(f"未知模板: {template}")
        
        return full_text
    
    def _find_response_start(self, tokens: List[int], item: Dict, template: str) -> int:
        """
        找到回复部分在token序列中的起始位置
        用于后续mask（只计算回复部分的损失）
        """
        # 编码提示部分（不含回复）
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        
        if template == "alpaca":
            if input_text:
                prompt_text = (
                    f"Below is an instruction that describes a task, paired with an input that provides further context. "
                    f"Write a response that appropriately completes the request.\\n\\n"
                    f"### Instruction:\\n{instruction}\\n\\n"
                    f"### Input:\\n{input_text}\\n\\n"
                    f"### Response:\\n"
                )
            else:
                prompt_text = (
                    f"Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request.\\n\\n"
                    f"### Instruction:\\n{instruction}\\n\\n"
                    f"### Response:\\n"
                )
        else:
            prompt_text = f"指令: {instruction}\\n输入: {input_text}\\n回复: "
        
        prompt_tokens = self.tokenizer.encode(prompt_text, allowed_special=set())
        
        # 回复起始位置（考虑BPE编码可能不完全匹配）
        # 简化处理：取min(len(prompt_tokens), len(tokens)-1)
        return min(len(prompt_tokens), len(tokens) - 1)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = sample["tokens"]
        response_start = sample["response_start"]
        
        # 填充到max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.eot_token] * (self.max_length - len(tokens))
        
        # 创建loss mask：回复部分为1，其他为0
        if self.mask_inputs:
            # 只计算回复部分的损失
            loss_mask = [0] * len(tokens)
            for i in range(response_start, len(tokens)):
                loss_mask[i] = 1
        else:
            # 计算全部损失
            loss_mask = [1] * len(tokens)
        
        # 输入和目标（右移一位）
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 和y对齐
        
        return {
            "input_ids": x,
            "labels": y,
            "loss_mask": mask,  # 关键：用于屏蔽输入部分的损失
            "response_start": response_start,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,  # Windows建议0，Linux/Mac可用4
    pin_memory: bool = True,
    collate_fn=None,
) -> DataLoader:
    """
    创建数据加载器
    
    参数:
        num_workers: 多进程加载数。Windows建议0（避免多进程问题），Linux/Mac可用4
    """
    # 自动选择collate_fn
    if collate_fn is None:
        from .collate_fn import pad_collate_fn
        collate_fn = pad_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,  # 避免单样本批次
        collate_fn=collate_fn,
    )


def get_openwebtext_dataloader(
    data_dir: str = "data",
    batch_size: int = 12,
    block_size: int = 1024,
    num_workers: int = 0,
) -> DataLoader:
    """
    获取OpenWebText风格的数据加载器
    如果本地不存在，创建示例数据用于测试
    """
    import os
    
    os.makedirs(data_dir, exist_ok=True)
    sample_file = os.path.join(data_dir, "sample.txt")
    
    # 创建示例数据（实际项目应下载真实数据）
    if not os.path.exists(sample_file):
        print("创建示例数据...")
        sample_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, 
as opposed to the natural intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, 
which refers to any system that perceives its environment and takes actions 
that maximize its chance of achieving its goals. The term "artificial intelligence" 
had previously been used to describe machines that mimic and display "human" 
cognitive skills that are associated with the human mind, such as "learning" 
and "problem-solving". This definition has since been rejected by major AI 
researchers who now describe AI in terms of rationality and acting rationally, 
which does not limit how intelligence can be articulated. 
""" * 100  # 重复以创建足够数据
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"示例数据已创建: {sample_file}")
    
    # 创建数据集
    dataset = TextDataset(
        data=sample_file,
        is_file=True,
        block_size=block_size,
    )
    
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def create_instruction_dataloader(
    data_path: str,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 0,
) -> DataLoader:
    """便捷函数：创建指令数据加载器"""
    dataset = InstructionDataset(
        data=data_path,
        max_length=max_length,
        mask_inputs=True,  # 关键：只计算回复损失
    )
    
    from .collate_fn import instruction_collate_fn
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=instruction_collate_fn,
    )