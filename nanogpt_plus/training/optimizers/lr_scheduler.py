"""
学习率调度器
支持：Cosine with Warmup（余弦退火+线性预热）
"""

import math


class CosineWarmupScheduler:
    """
    余弦退火 + 线性预热学习率调度
    
    训练过程：
    1. Warmup阶段（前warmup_iters步）：从0线性增长到max_lr
    2. Cosine阶段：余弦下降到min_lr
    
    为什么需要Warmup？
    - 训练初期梯度大，大学习率会导致震荡
    - 先小步摸索，再大步前进
    
    为什么Cosine退火？
    - 后期精细调整，避免错过最优解
    - 比StepDecay（阶梯下降）更平滑
    """
    
    def __init__(
        self,
        optimizer,           # PyTorch优化器
        warmup_iters: int,  # 预热步数（如2000）
        max_iters: int,     # 总训练步数（如100000）
        max_lr: float,      # 最大学习率（如6e-4）
        min_lr: float = 0.0, # 最小学习率（如6e-5）
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # 记录每个参数组的初始学习率（支持不同组不同学习率）
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, iter_num: int) -> float:
        """
        根据当前迭代数更新学习率
        
        返回：当前学习率
        """
        if iter_num < self.warmup_iters:
            # 线性预热：从0增长到max_lr
            lr_mult = iter_num / self.warmup_iters
        else:
            # 余弦退火
            progress = (iter_num - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            progress = min(1.0, progress)  # 防止超调
            
            # 余弦系数：从1降到0
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # 学习率 = min_lr + (max_lr - min_lr) * cosine_decay
            lr_mult = self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * cosine_decay
        
        # 应用到所有参数组
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * lr_mult
        
        return self.optimizer.param_groups[0]['lr']
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self) -> dict:
        """保存状态（用于恢复训练）"""
        return {
            'base_lrs': self.base_lrs,
            'warmup_iters': self.warmup_iters,
            'max_iters': self.max_iters,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
        }
    
    def load_state_dict(self, state_dict: dict):
        """加载状态"""
        self.base_lrs = state_dict['base_lrs']
        self.warmup_iters = state_dict['warmup_iters']
        self.max_iters = state_dict['max_iters']
        self.max_lr = state_dict['max_lr']
        self.min_lr = state_dict['min_lr']
