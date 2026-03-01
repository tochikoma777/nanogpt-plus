"""
模型模块
包含GPT架构和各种变体（LoRA、MoE等）
"""

from .gpt import GPT, create_model, LayerNorm, CausalSelfAttention, Block
from .lora import LoRALinear, inject_lora, get_lora_params, print_trainable_parameters

__all__ = [
    'GPT',
    'create_model',
    'LayerNorm',
    'CausalSelfAttention',
    'Block',
    'LoRALinear',
    'inject_lora',
    'get_lora_params',
    'print_trainable_parameters',
]
