"""
模型单元测试
"""

import torch
import pytest

from nanogpt_plus.config import ModelConfig
from nanogpt_plus.models.gpt import create_model, GPT
from nanogpt_plus.models.lora import freeze_non_lora_params, inject_lora, get_lora_params


def test_model_creation():
    """测试模型创建"""
    config = ModelConfig(
        vocab_size=100,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    model = create_model(config)
    assert isinstance(model, GPT)
    assert model.get_num_params() > 0


def test_forward_pass():
    """测试前向传播"""
    config = ModelConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    model = create_model(config)
    
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(x)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None  # 没有提供targets


def test_forward_with_loss():
    """测试带损失的前向传播"""
    config = ModelConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    model = create_model(config)
    
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_generation():
    """测试文本生成"""
    config = ModelConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    model = create_model(config)
    model.eval()
    
    # 初始序列
    idx = torch.randint(0, config.vocab_size, (1, 5))
    
    # 生成
    generated = model.generate(idx, max_new_tokens=10, temperature=1.0)
    assert generated.shape == (1, 5 + 10)


def test_lora_injection():
    """测试LoRA注入"""
    config = ModelConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    model = create_model(config)
    
    # 记录原始参数量
    total_params_before = sum(p.numel() for p in model.parameters())
    
    # 注入LoRA
    inject_lora(model, r=4, lora_alpha=8)
    freeze_non_lora_params(model)
     
    # 检查LoRA参数存在
    lora_params = get_lora_params(model)
    assert len(lora_params) > 0
    
    # 检查可训练参数比例很低
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total
    assert ratio < 0.1  # 应小于10%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
