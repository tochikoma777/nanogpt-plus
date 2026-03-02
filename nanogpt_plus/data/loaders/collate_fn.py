"""
Collate函数：将多个样本组合成一个批次
处理变长序列的填充（padding）和mask
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate_fn(batch: list) -> dict:
    """
    基础填充函数
    将变长序列填充到批次内最大长度
    """
    # 分离字段
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    lengths = [item.get("length", len(x)) for item, x in zip(batch, input_ids)]
    
    # 填充（使用0，但后面会被loss_mask处理）
    # pad_sequence默认batch_first=False，需要指定
    x_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    y_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100被cross_entropy忽略
    
    # 创建attention mask（实际不需要，因为是因果模型）
    # 但可用于其他用途
    mask = torch.ones_like(x_padded, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, length:] = False
    
    return {
        "input_ids": x_padded,
        "labels": y_padded,
        "attention_mask": mask,
    }


def instruction_collate_fn(batch: list) -> dict:
    """
    指令数据的collate函数
    关键：处理loss_mask，只计算回复部分的损失
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    loss_masks = [item["loss_mask"] for item in batch]
    
    # 填充
    x_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    y_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    mask_padded = pad_sequence(loss_masks, batch_first=True, padding_value=0)
    
    # 应用mask到labels：mask=0的位置设为-100（忽略）
    # 这样cross_entropy不会计算这些位置的损失
    y_masked = y_padded.clone()
    y_masked[mask_padded == 0] = -100
    
    return {
        "input_ids": x_padded,
        "labels": y_masked,  # 关键：已应用mask
        "loss_mask": mask_padded,  # 保留用于调试
    }


def dpo_collate_fn(batch: list) -> dict:
    """
    DPO（直接偏好优化）数据的collate函数
    包含chosen（偏好）和rejected（不偏好）两个回复
    """
    # 分离两个回复的数据
    chosen_input_ids = [item["chosen_input_ids"] for item in batch]
    chosen_labels = [item["chosen_labels"] for item in batch]
    rejected_input_ids = [item["rejected_input_ids"] for item in batch]
    rejected_labels = [item["rejected_labels"] for item in batch]
    
    # 分别填充
    chosen_x = pad_sequence(chosen_input_ids, batch_first=True, padding_value=0)
    chosen_y = pad_sequence(chosen_labels, batch_first=True, padding_value=-100)
    rejected_x = pad_sequence(rejected_input_ids, batch_first=True, padding_value=0)
    rejected_y = pad_sequence(rejected_labels, batch_first=True, padding_value=-100)
    
    return {
        "chosen_input_ids": chosen_x,
        "chosen_labels": chosen_y,
        "rejected_input_ids": rejected_x,
        "rejected_labels": rejected_y,
    }