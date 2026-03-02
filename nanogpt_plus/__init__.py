"""
NanoGPT-Plus: 轻量级、可扩展的GPT训练框架
"""

__version__ = "0.1.0"

# 配置
from .config import ModelConfig, TrainingConfig, DataConfig

# 模型
from .models import (
    GPT,
    create_model,
    LayerNorm,
    CausalSelfAttention,
    Block,
    LoRALinear,
    inject_lora,
    get_lora_params,
    print_trainable_parameters,
    setup_lora_for_training,
)

# 数据
from .data import (
    TextDataset,
    InstructionDataset,
    create_dataloader,
    instruction_collate_fn,
)

# 训练
from .training import Trainer, CosineWarmupScheduler

# 工具
from .utils import save_checkpoint, load_checkpoint, get_logger


__all__ = [
    # 版本
    "__version__",
    # 配置
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # 模型
    "GPT",
    "create_model",
    "LayerNorm",
    "CausalSelfAttention",
    "Block",
    "LoRALinear",
    "inject_lora",
    "get_lora_params",
    "print_trainable_parameters",
    "setup_lora_for_training",
    # 数据
    "TextDataset",
    "InstructionDataset",
    "create_dataloader",
    "instruction_collate_fn",
    # 训练
    "Trainer",
    "CosineWarmupScheduler",
    # 工具
    "save_checkpoint",
    "load_checkpoint",
    "get_logger",
]
