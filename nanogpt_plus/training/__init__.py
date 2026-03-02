"""
训练模块
包含Trainer、优化器、学习率调度、回调函数
"""

from .trainer import Trainer
from .optimizers.lr_scheduler import CosineWarmupScheduler

__all__ = ['Trainer', 'CosineWarmupScheduler']
