# NanoGPT-Plus 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**轻量级、可扩展、生产就绪的GPT训练框架**

NanoGPT，实现模块化架构、LoRA微调、DPO对齐（开发中）、量化推理（开发中）等优化特性。

---

## ✨ 核心特性

### 🏗️ 模块化架构

- **可插拔设计**：模型、训练策略、数据加载器均可独立替换
- **配置驱动**：YAML配置，实验可复现
- **类型安全**：完整类型注解，IDE友好

### 🚀 高效训练(开发中)

- **Flash Attention**：自动检测并使用，速度提升2-3倍
- **混合精度**：BF16/FP16支持，节省显存
- **分布式**：DDP/FSDP/DeepSpeed无缝切换
- **梯度优化**：累积、裁剪、检查点，训练大模型更稳

### 🎯 参数高效微调

- **LoRA/QLoRA**：只训练0.1%参数，显存降低80%
- **自动注入**：一行代码替换目标层
- **权重合并**：推理时合并，零开销

### ⚖️ 对齐训练（开发中）

- **SFT**：指令微调，支持Alpaca/ShareGPT格式
- **DPO**：直接偏好优化，无需奖励模型
- **RLHF**：完整PPO流程（即将支持）

### 🔋 推理优化（开发中）

- **量化**：GPTQ/AWQ 4bit推理
- **投机采样**：小模型起草+大模型验证，速度提升2倍
- **vLLM集成**：生产级推理服务

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/tochikoma777/nanogpt-plus.git
cd nanogpt-plus

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -e ".[dev]"
```

### 验证安装

```bash
python -c "import nanogpt_plus; print('安装成功！')"
pytest tests/unit/test_model.py -v  # 运行单元测试
```

### 1分钟上手：从头训练

```python
from nanogpt_plus import ModelConfig, TrainingConfig, create_model, Trainer
from nanogpt_plus.data import get_openwebtext_dataloader

# 配置
model_cfg = ModelConfig(n_layer=6, n_embd=384, n_head=6)  # 小型模型
train_cfg = TrainingConfig(max_iters=5000, batch_size=8)

# 创建模型和数据
model = create_model(model_cfg)
train_loader = get_openwebtext_dataloader(batch_size=8)

# 训练
trainer = Trainer(model, train_cfg, train_loader)
trainer.train()
```

### LoRA微调示例

```python
from nanogpt_plus.models import inject_lora, print_trainable_parameters

# 加载预训练模型
model = create_model(ModelConfig(n_layer=12, n_embd=768))

# 注入LoRA（只训练这些低秩矩阵）
inject_lora(model, r=16, target_modules=["c_attn", "c_proj"])
print_trainable_parameters(model)  # 显示可训练参数占比（通常<1%）

# 正常训练，显存占用大幅降低
```

---

### 📁 项目结构

```
nanogpt-plus/
├── nanogpt_plus/          # 主包
│   ├── models/            # GPT架构 + LoRA/MoE等变体
│   ├── training/          # 训练器 + 分布式策略 + 优化器
│   ├── data/              # 数据加载器（文本/指令/DPO）
│   ├── inference/         # 生成 + 量化 + 加速
│   └── utils/             # 工具函数
├── configs/               # YAML配置文件（Hydra）
├── examples/              # 可运行示例
├── tests/                 # 单元测试 + 集成测试
└── docs/                  # 文档
```

---

### 📊 性能基准

待获取GPU/TPU资源后开展全面测试

*测试条件：batch_size=16, sequence_length=1024, 混合精度bf16*

---

### 📚 教程目录

1. **[从头训练GPT](examples/01_train_from_scratch.py)** - 理解训练循环
2. **[LoRA微调](examples/02_finetune_with_lora.py)** - 参数高效微调
3. **[DPO对齐](examples/03_dpo_alignment.py)** - 人类偏好对齐（即将添加）
4. **[量化部署](examples/04_quantize_and_deploy.py)** - 4bit推理（即将添加）

---

### 🛠️ 开发指南

#### 代码风格

```bash
# 格式化代码
black nanogpt_plus/ examples/ tests/
isort nanogpt_plus/ examples/ tests/

# 类型检查
mypy nanogpt_plus/

# 运行测试
pytest tests/ -v --cov=nanogpt_plus
```

#### 添加新模型

```python
# nanogpt_plus/models/my_model.py
from nanogpt_plus.config import ModelConfig

class MyModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 你的实现
    
    def forward(self, x):
        # 前向传播
        pass

# 在__init__.py中注册
from .my_model import MyModel
```

---

### 🗺️ 路线图

- [x] v0.1.0: 核心GPT + LoRA + 基础训练
- [ ] v0.2.0: DPO训练 + 多轮对话 + 中文支持
- [ ] v0.3.0: 量化推理 + 投机采样 + vLLM
- [ ] v0.4.0: MoE架构 + 长上下文扩展
- [ ] v1.0.0: 稳定API + 完整文档 + 社区生态

---

### 🤝 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

特别需要：

- 🐛 Bug修复和性能优化
- 📖 文档和教程（目前仅支持中文）
- 🌍 多语言Tokenizer支持
- 🧪 更多评测数据集

---

### 🙏 致谢

- 首先感谢自己的坚持和努力😀

---

### 📄 许可证

Apache 2.0 - 可自由用于商业和研究。

**Star ⭐ 如果这个项目对你有帮助！**# nanogpt-plus
轻量级、可扩展的GPT训练框架，支持LoRA/DPO/量化推理
