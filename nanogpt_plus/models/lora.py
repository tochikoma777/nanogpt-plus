"""
LoRA: Low-Rank Adaptation of Large Language Models
参数高效微调，只训练低秩矩阵，冻结原权重

论文: https://arxiv.org/abs/2106.09685
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA线性层
    
    替换标准的nn.Linear，冻结原权重，只训练低秩分解矩阵A和B
    
    数学原理：
        h = Wx + BAx
        其中：
        - W: 原权重矩阵，形状 (out_features, in_features)，冻结
        - A: 低秩矩阵，形状 (in_features, r)，可训练，高斯初始化
        - B: 低秩矩阵，形状 (r, out_features)，可训练，零初始化
        - r: 秩（rank），通常8-64，远小于min(in_features, out_features)
        - BA的秩最多为r，因此称为"低秩"适配
    
    初始化策略：
        - A用高斯噪声初始化（打破对称性）
        - B用零初始化（确保训练开始时BA=0，不改变原输出）
        这样训练初期模型表现和预训练一致，逐渐学习新任务
    """
    
    def __init__(
        self,
        in_features: int,          # 输入维度（如768）
        out_features: int,         # 输出维度（如768）
        r: int = 8,                 # LoRA秩，控制可训练参数量
        lora_alpha: int = 16,       # 缩放因子，控制LoRA影响强度
        lora_dropout: float = 0.0,  # LoRA路径的dropout（防止过拟合）
        merge_weights: bool = False, # 推理时是否合并权重（加速）
        bias: bool = True,          # 是否使用偏置
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r                          # 秩
        self.lora_alpha = lora_alpha        # 缩放因子
        self.scaling = lora_alpha / r       # 实际缩放比例
        self.merge_weights = merge_weights  # 是否自动合并
        self.merged = False                 # 当前是否已合并
        
        # 验证：秩不能超过输入/输出维度
        if r > min(in_features, out_features):
            raise ValueError(f"LoRA rank {r} > min({in_features}, {out_features})")
        
        # ========== 原权重（冻结） ==========
        # 创建但不训练，用于存储预训练权重
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features), 
            requires_grad=False  # ❌ 关键：不计算梯度
        )
        
        # 偏置（如果有）
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features),
                requires_grad=False  # 同样冻结
            )
        else:
            self.register_parameter('bias', None)
        
        # ========== LoRA可训练参数 ==========
        # A矩阵：in_features × r（瘦高矩阵）
        # 用标准正态分布初始化（均值为0，标准差为0.02）
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        
        # B矩阵：r × out_features（矮胖矩阵）
        # 用零初始化（关键！确保训练初期不改变输出）
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # LoRA路径的dropout（可选）
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化所有参数"""
        # 原权重：标准初始化（复制预训练权重时会覆盖）
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # LoRA A：高斯初始化（打破对称，让不同维度学到不同特征）
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        
        # LoRA B：零初始化（关键！训练开始时BA=0）
        nn.init.zeros_(self.lora_B)
    
    def train(self, mode: bool = True):
        """
        切换训练/评估模式
        
        特殊处理：如果merge_weights=True，在评估时自动合并权重加速推理
        """
        super().train(mode)
        
        if mode:
            # 训练模式：如果之前合并了，现在解合并
            if self.merge_weights and self.merged:
                # 从权重中减去LoRA部分
                # 注意：需要转置因为PyTorch线性层是(out, in)
                self.weight.data -= (self.lora_B @ self.lora_A.T).T * self.scaling
                self.merged = False
        else:
            # 评估模式：如果要求合并且未合并，则合并
            if self.merge_weights and not self.merged:
                # 把LoRA权重加到原权重上
                self.weight.data += (self.lora_B @ self.lora_A.T).T * self.scaling
                self.merged = True
        
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        计算：output = Wx + scaling * BAx
        
        如果已合并（merged=True），则简化为：output = Wx（更快）
        """
        # 原权重路径（始终计算，但冻结）
        original = F.linear(x, self.weight, self.bias)
        
        # LoRA路径（只在训练时或有梯度时计算）
        if self.r > 0 and not self.merged:
            # x: (batch, seq_len, in_features)
            # lora_dropout(x): 随机丢弃部分输入（正则化）
            # @ lora_A: (batch, seq_len, in_features) @ (in_features, r) -> (batch, seq_len, r)
            # @ lora_B: (batch, seq_len, r) @ (r, out_features) -> (batch, seq_len, out_features)
            lora = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
            
            return original + lora
        
        # 已合并时，直接返回原权重输出（更快）
        return original
    
    def merge(self) -> nn.Linear:
        """
        手动合并LoRA权重到原权重，返回标准Linear层
        
        用途：训练完成后，合并权重用于高效推理
        """
        if self.merged:
            print("权重已经合并")
            return self
        
        # 创建新的Linear层
        merged_linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        
        # 复制合并后的权重
        merged_weight = self.weight.data + (self.lora_B @ self.lora_A.T).T * self.scaling
        merged_linear.weight.data.copy_(merged_weight)
        
        if self.bias is not None:
            merged_linear.bias.data.copy_(self.bias.data)
        
        return merged_linear
    
    def get_lora_params(self) -> List[torch.nn.Parameter]:
        """获取LoRA可训练参数"""
        return [self.lora_A, self.lora_B]
    
    def __repr__(self):
        return (
            f"LoRALinear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"r={self.r}, "
            f"lora_alpha={self.lora_alpha}, "
            f"scaling={self.scaling:.3f})"
        )


def inject_lora(
    model: nn.Module,
    target_modules: List[str] = ["c_attn", "c_proj", "c_fc"],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    merge_weights: bool = False,
) -> None:
    """
    自动替换模型中的目标线性层为LoRA版本
    
    参数:
        model: 要修改的模型（如GPT）
        target_modules: 要替换的模块名列表
            - "c_attn": 注意力QKV投影
            - "c_proj": 注意力输出投影
            - "c_fc": MLP第一层（扩展）
            - "c_proj" in MLP: MLP第二层（压缩）
        r: LoRA秩（默认8）
        lora_alpha: 缩放因子（默认16，scaling=2）
        lora_dropout: LoRA路径dropout率
        merge_weights: 是否自动合并（推理优化）
    
    示例:
        model = create_model(ModelConfig())
        inject_lora(model, r=16, target_modules=["c_attn", "c_proj"])
        # 现在模型中的目标层已被替换为LoRALinear
    """
    replaced_count = 0
    
    # 遍历所有命名模块
    for name, module in model.named_modules():
        # 检查模块名是否匹配目标
        # 例如：匹配"h.0.attn.c_attn"中的"c_attn"
        if any(target in name for target in target_modules):
            # 找到父模块和当前模块在父模块中的属性名
            # 例如：name="h.0.attn.c_attn" -> parent="h.0.attn", child_name="c_attn"
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            # 获取父模块
            if parent_name == "":
                parent = model
            else:
                parent = model.get_submodule(parent_name)
            
            # 只替换nn.Linear（不替换已经替换过的）
            if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                # 创建LoRA层
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=merge_weights,
                    bias=module.bias is not None,
                )
                
                # 复制原权重（冻结）
                lora_layer.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    lora_layer.bias.data.copy_(module.bias.data)
                
                # 替换：把父模块的对应属性设为新的LoRA层
                setattr(parent, child_name, lora_layer)
                replaced_count += 1
                
                print(f"  ✓ 替换 {name}: {module.in_features}x{module.out_features} -> LoRA(r={r})")
    
    print(f"\n总共替换了 {replaced_count} 个层为LoRA")


def get_lora_params(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    获取模型中所有LoRA可训练参数
    
    用于创建优化器（只优化这些参数）
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:  # 匹配lora_A和lora_B
            lora_params.append(param)
    return lora_params


def freeze_non_lora_params(model: nn.Module) -> None:
    """
    冻结所有非LoRA参数
    
    通常在inject_lora后调用
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def print_trainable_parameters(model: nn.Module) -> None:
    """
    打印可训练参数统计信息
    
    用于验证LoRA是否生效（可训练参数应<<总参数）
    """
    trainable_params = 0
    all_param = 0
    lora_param = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora_" in name:
                lora_param += param.numel()
    
    # 计算百分比
    trainable_ratio = 100 * trainable_params / all_param
    lora_ratio = 100 * lora_param / all_param
    
    print(f"\n{'='*50}")
    print(f"参数统计:")
    print(f"  总参数量:       {all_param:,} ({all_param/1e6:.2f}M)")
    print(f"  可训练参数:     {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"    └─ LoRA参数:  {lora_param:,} ({lora_param/1e6:.2f}M)")
    print(f"  冻结参数:       {all_param - trainable_params:,}")
    print(f"\n可训练占比:       {trainable_ratio:.4f}%")
    print(f"LoRA占比:         {lora_ratio:.4f}%")
    print(f"{'='*50}")
    
    # 验证：如果LoRA生效，占比应该很小（<1%）
    if trainable_ratio > 5:
        print("⚠️ 警告: 可训练参数占比>5%，请检查是否冻结了原权重")
    else:
        print("✅ LoRA配置正确，参数高效！")


def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    只保存LoRA权重（小文件，便于分享）
    
    文件大小通常只有几MB，而非LoRA模型的几百MB-几GB
    """
    lora_state = {
        name: param.cpu().clone()
        for name, param in model.named_items()
        if "lora_" in name
    }
    
    torch.save(lora_state, path)
    print(f"LoRA权重已保存: {path}")
    print(f"文件大小: {sum(p.numel() for p in lora_state.values()) * 4 / 1024 / 1024:.2f} MB")


def load_lora_weights(model: nn.Module, path: str) -> None:
    """加载LoRA权重到模型"""
    lora_state = torch.load(path, map_location="cpu")
    
    # 加载到模型
    model_state = model.state_dict()
    model_state.update(lora_state)
    model.load_state_dict(model_state)
    
    print(f"LoRA权重已加载: {path}")


# 便捷函数：一键LoRA设置
def setup_lora_for_training(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: List[str] = ["c_attn", "c_proj"],
    lora_dropout: float = 0.05,
) -> List[torch.nn.Parameter]:
    """
    一键配置LoRA用于训练
    
    包含：注入LoRA、冻结原权重、打印统计、返回可训练参数
    
    返回:
        可训练的LoRA参数列表（用于创建优化器）
    
    示例:
        model = create_model(config)
        lora_params = setup_lora_for_training(model, r=16)
        optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    """
    print("=" * 50)
    print("设置LoRA用于训练")
    print("=" * 50)
    
    # 1. 注入LoRA
    print(f"\n1. 注入LoRA (r={r}, alpha={lora_alpha})...")
    inject_lora(
        model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    # 2. 冻结非LoRA参数
    print(f"\n2. 冻结原权重...")
    freeze_non_lora_params(model)
    
    # 3. 打印统计
    print(f"\n3. 参数统计:")
    print_trainable_parameters(model)
    
    # 4. 返回可训练参数
    lora_params = get_lora_params(model)
    print(f"\n4. 准备好 {len(lora_params)} 个LoRA参数用于训练")
    print("=" * 50)
    
    return lora_params