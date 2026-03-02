"""
数据模块
支持通用文本、指令微调、DPO偏好数据
"""

from .loaders.text_loader import (
    TextDataset,
    InstructionDataset,
    create_dataloader,
    get_openwebtext_dataloader,
)
from .loaders.collate_fn import instruction_collate_fn, pad_collate_fn

__all__ = [
    'TextDataset',
    'InstructionDataset',
    'create_dataloader',
    'get_openwebtext_dataloader',
    'instruction_collate_fn',
    'pad_collate_fn',
]