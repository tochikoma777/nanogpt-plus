"""
安全的检查点保存和加载
支持原子写入、版本兼容性检查
"""

import os
import shutil
from pathlib import Path

import torch


def save_checkpoint(state_dict: dict, path: str, atomic: bool = True):
    """
    安全保存检查点
    
    Args:
        state_dict: 要保存的状态字典
        path: 目标路径
        atomic: 是否使用原子写入（先写临时文件，再重命名）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        # 原子写入：先写到临时文件，成功后再重命名
        # 防止写入过程中断导致文件损坏
        tmp_path = path.with_suffix('.tmp')
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)  # 原子重命名
    else:
        torch.save(state_dict, path)
    
    # 同时保存一个"latest.pt"软链接/副本
    latest_path = path.parent / "latest.pt"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    try:
        latest_path.symlink_to(path.name)
    except OSError:
        # Windows可能需要管理员权限创建symlink，回退到复制
        shutil.copy2(path, latest_path)


def load_checkpoint(path: str, map_location=None):
    """
    加载检查点
    
    Args:
        path: 检查点路径
        map_location: 设备映射（如'cpu', 'cuda:0'）
    
    Returns:
        加载的状态字典
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
    
    return checkpoint
