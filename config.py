# 确保在nanogpt_plus目录
cd nanogpt_plus

# 创建config.py
cat > config.py << 'EOF'
"""
统一配置管理
支持从YAML文件加载和命令行覆盖
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """模型架构配置"""
    name: str = "gpt2-small"
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False
    
    @property
    def head_dim(self) -> int:
        """每个头的维度（自动计算）"""
        return self.n_embd // self.n_head


@dataclass
class TrainingConfig:
    """训练配置"""
    max_iters: int = 100000
    batch_size: int = 12
    gradient_accumulation_steps: int = 1
    block_size: int = 1024
    
    # 优化器
    learning_rate: float = 6.0e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # 调度器
    lr_scheduler: str = "cosine"
    warmup_iters: int = 2000
    min_lr: float = 6.0e-5
    
    # 日志与保存
    eval_interval: int = 1000
    eval_iters: int = 200
    log_interval: int = 10
    checkpoint_interval: int = 5000
    out_dir: str = "outputs"
    
    # 系统
    device: str = "cuda"
    compile: bool = True
    dtype: str = "bfloat16"
    seed: int = 1337


@dataclass
class DataConfig:
    """数据配置"""
    name: str = "openwebtext"
    dataset_path: str = "openwebtext"
    tokenizer: str = "gpt2"
    text_column: str = "text"
    block_size: int = 1024
    stride: int = 1024
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


def load_config(config_path: str) -> dict:
    """
    从YAML文件加载配置
    
    用法:
        config = load_config("configs/model/gpt2_small.yaml")
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: Optional[dict] = None) -> dict:
    """
    合并配置，override优先级更高
    
    用法:
        # 基础配置 + 实验特定覆盖
        config = merge_configs(base_config, {"learning_rate": 1e-4})
    """
    if override is None:
        return base
    result = base.copy()
    result.update(override)
    return result


# 便捷函数：直接加载为dataclass
def load_model_config(path: str) -> ModelConfig:
    """从YAML加载模型配置"""
    data = load_config(path)
    return ModelConfig(**data)


def load_training_config(path: str) -> TrainingConfig:
    """从YAML加载训练配置"""
    data = load_config(path)
    return TrainingConfig(**data)


def load_data_config(path: str) -> DataConfig:
    """从YAML加载数据配置"""
    data = load_config(path)
    return DataConfig(**data)
EOF