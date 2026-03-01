"""
模型模块
包含GPT架构和各种变体（LoRA、MoE等）
"""

from .gpt import (
    GPT, 
    create_model, 
    LayerNorm, 
    CausalSelfAttention, 
    Block, 
    MLP
)

from .lora import (
    LoRALinear, 
    inject_lora, 
    get_lora_params, 
    print_trainable_parameters,
    freeze_non_lora_params,
    setup_lora_for_training,
    save_lora_weights,
    load_lora_weights,
)

__all__ = [
    # GPT核心
    'GPT',
    'create_model',
    'LayerNorm',
    'CausalSelfAttention',
    'Block',
    'MLP',
    # LoRA
    'LoRALinear',
    'inject_lora',
    'get_lora_params',
    'print_trainable_parameters',
    'freeze_non_lora_params',
    'setup_lora_for_training',
    'save_lora_weights',
    'load_lora_weights',
]
