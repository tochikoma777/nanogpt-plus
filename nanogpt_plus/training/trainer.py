"""
通用训练器（Trainer）
封装训练循环、评估、保存、日志等逻辑
"""

import os
import time
import math
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from nanogpt_plus.config import TrainingConfig
from nanogpt_plus.training.optimizers.lr_scheduler import CosineWarmupScheduler
from nanogpt_plus.utils.checkpoint_io import save_checkpoint, load_checkpoint
from nanogpt_plus.utils.logging import get_logger


# 修复FutureWarning：使用新的torch.amp API
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    # 兼容旧版本
    from torch.cuda.amp import GradScaler, autocast


logger = get_logger(__name__)


class Trainer:
    """
    通用训练器
    
    功能：
    1. 训练循环（支持梯度累积、混合精度）
    2. 评估验证集
    3. 学习率调度（Cosine + Warmup）
    4. 检查点保存/加载（断点续训）
    5. 日志记录（TensorBoard + 控制台）
    6. 梯度裁剪（防止爆炸）
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 设备设置
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_iters=config.warmup_iters,
            max_iters=config.max_iters,
            max_lr=config.learning_rate,
            min_lr=config.min_lr,
        )
        
        # 混合精度设置
        self.dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(config.dtype, torch.bfloat16)
        
        self.use_amp = config.dtype in ["float16", "bfloat16"]
        self.scaler = GradScaler(enabled=(config.dtype == "float16"))
        
        # PyTorch 2.0编译优化
        if config.compile and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # 训练状态
        self.iter_num = 0
        self.best_val_loss = float('inf')
        self.running_loss = 0.0
        
        # 输出目录
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard日志
        self.writer = SummaryWriter(self.out_dir / "tensorboard")
        
        # 尝试加载已有检查点
        self._load_checkpoint_if_exists()
        
        logger.info(f"Trainer initialized: device={self.device}, dtype={config.dtype}")
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练（支持梯度累积）"""
        self.model.train()
        
        # 数据移动到设备
        x = batch["input_ids"].to(self.device)
        y = batch["labels"].to(self.device)
        
        # 混合精度上下文
        with autocast(enabled=self.use_amp, dtype=self.dtype):
            logits, loss = self.model(x, y)
            loss = loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        if self.use_amp and self.dtype == torch.float16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self) -> float:
        """执行参数更新"""
        # 梯度裁剪
        if self.config.grad_clip > 0:
            if self.use_amp and self.dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.grad_clip
            )
        
        # 参数更新
        if self.use_amp and self.dtype == torch.float16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # 清零梯度
        self.optimizer.zero_grad(set_to_none=True)
        
        # 更新学习率
        lr = self.scheduler.step(self.iter_num)
        return lr
    
    def evaluate(self) -> float:
        """评估验证集"""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.config.eval_iters:
                    break
                
                x = batch["input_ids"].to(self.device)
                y = batch["labels"].to(self.device)
                
                with autocast(enabled=self.use_amp, dtype=self.dtype):
                    logits, loss = self.model(x, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.model.train()
        return avg_loss
    
    def train(self) -> None:
        """主训练循环"""
        logger.info(f"Starting training for {self.config.max_iters} iterations")
        
        t0 = time.time()
        train_iter = iter(self.train_loader)
        
        while self.iter_num < self.config.max_iters:
            # 梯度累积循环
            accumulated_loss = 0.0
            
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                
                loss = self.training_step(batch)
                accumulated_loss += loss
            
            # 参数更新
            lr = self.optimizer_step()
            
            # 更新统计
            self.iter_num += 1
            self.running_loss = 0.9 * self.running_loss + 0.1 * accumulated_loss
            
            # 日志记录
            if self.iter_num % self.config.log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                
                tokens_per_sec = (
                    self.config.batch_size * 
                    self.config.block_size * 
                    self.config.gradient_accumulation_steps
                ) / dt
                
                perplexity = math.exp(self.running_loss)
                
                # 控制台输出
                msg = (
                    f"Iter {self.iter_num:6d}/{self.config.max_iters} | "
                    f"Loss: {accumulated_loss:.4f} | "
                    f"PPL: {perplexity:.2f} | "
                    f"LR: {lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f}"
                )
                logger.info(msg)
                
                # TensorBoard记录
                self.writer.add_scalar("train/loss", accumulated_loss, self.iter_num)
                self.writer.add_scalar("train/perplexity", perplexity, self.iter_num)
                self.writer.add_scalar("train/lr", lr, self.iter_num)
            
            # 评估验证集
            if self.iter_num % self.config.eval_interval == 0 and self.val_loader:
                val_loss = self.evaluate()
                val_perplexity = math.exp(val_loss)
                
                logger.info(f"Validation | Loss: {val_loss:.4f} | PPL: {val_perplexity:.2f}")
                
                self.writer.add_scalar("val/loss", val_loss, self.iter_num)
                self.writer.add_scalar("val/perplexity", val_perplexity, self.iter_num)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(is_best=True)
            
            # 定期保存
            if self.iter_num % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
        
        # 训练结束
        self._save_checkpoint(is_final=True)
        self.writer.close()
        
        logger.info(f"Training completed! Best val loss: {self.best_val_loss:.4f}")
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """保存检查点"""
        if is_best:
            filename = "best.pt"
        elif is_final:
            filename = "final.pt"
        else:
            filename = f"iter_{self.iter_num:06d}.pt"
        
        path = self.out_dir / filename
        
        checkpoint = {
            "iter_num": self.iter_num,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        save_checkpoint(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def _load_checkpoint_if_exists(self):
        """加载最新的检查点（如果存在）"""
        latest_path = self.out_dir / "latest.pt"
        if not latest_path.exists():
            return
        
        logger.info(f"Resuming from checkpoint: {latest_path}")
        
        checkpoint = load_checkpoint(latest_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.iter_num = checkpoint["iter_num"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        logger.info(f"Resumed at iteration {self.iter_num}")
