"""
NanoGPT-Plus 核心模型实现
完整的GPT（Generative Pre-trained Transformer）架构
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from nanogpt_plus.config import ModelConfig


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）
    
    作用：稳定深层网络训练，让每一层的输出分布保持一致
    
    与普通LayerNorm的区别：可以控制是否使用bias（GPT-2不用bias）
    """
    
    def __init__(self, ndim: int, bias: bool = True):
        """
        参数:
            ndim: 输入特征的维度（如768）
            bias: 是否使用偏置项（GPT-2设为False）
        """
        super().__init__()
        # 可学习的缩放参数（gamma），初始为1（不改变）
        self.weight = nn.Parameter(torch.ones(ndim))
        # 可学习的偏移参数（beta），初始为0（不改变）
        # 如果bias=False，则不创建（节省参数）
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对最后一个维度做归一化
        
        输入形状: (batch_size, seq_len, ndim)
        输出形状: 相同
        """
        # F.layer_norm是PyTorch优化过的实现，比手动计算更快
        return F.layer_norm(
            input, 
            self.weight.shape,  # 归一化的维度
            self.weight,        # 缩放
            self.bias,          # 偏移（可能为None）
            1e-5                # 防止除以0的小数
        )


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）
    
    "因果" = 只能看当前和之前的token，不能偷看未来（自回归特性）
    "自" = Query/Key/Value都来自同一个输入
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 断言：确保维度能被头数整除
        assert config.n_embd % config.n_head == 0, "n_embd必须能被n_head整除"
        
        # 保存配置
        self.n_head = config.n_head      # 注意力头数（如12）
        self.n_embd = config.n_embd      # 总维度（如768）
        self.head_dim = config.head_dim  # 每头维度（768/12=64）
        self.dropout = config.dropout
        
        # 关键优化：合并QKV投影为一个大矩阵
        # 而不是三个独立层，减少kernel启动开销
        # 形状: (n_embd, 3*n_embd) = (768, 2304)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 输出投影：把注意力结果映射回n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 因果掩码：上三角为0（禁止看未来）
        # 注册为buffer（不是参数，不训练，但随模型保存）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        # 结果形状: (1, 1, block_size, block_size)
        # 前两个1是为了广播到(batch, n_head)
        
        # Flash Attention检测（PyTorch 2.0+内置）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("⚠️ 警告: Flash Attention不可用，使用标准实现（较慢）")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入x: (batch_size, seq_len, n_embd)
        输出: 相同形状
        """
        B, T, C = x.size()  # Batch, Time (seq_len), Channels (n_embd)
        
        # 步骤1: 计算QKV（合并投影后分割）
        # qkv: (B, T, 3*C) -> 分割为3个(B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 步骤2: 重塑为多头格式
        # 从 (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        # 这样每个头可以独立计算注意力
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # 现在形状: (B, n_head, T, head_dim)
        
        # 步骤3: 注意力计算
        if self.flash:
            # ✅ Flash Attention: 融合kernel，更快更省显存
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,           # 不需要显式mask
                dropout_p=self.dropout if self.training else 0,
                is_causal=True            # 自动处理因果掩码
            )
        else:
            # 手动实现（兼容旧PyTorch）
            # 注意力分数: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) 
            #         -> (B, n_head, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # 应用因果掩码：把未来位置设为-inf（softmax后变为0）
            # self.bias[:, :, :T, :T] 取出当前长度的子矩阵
            # == 0的位置表示"禁止看"
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            
            # Softmax归一化（每行之和为1）
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # 加权求和: (B, n_head, T, T) @ (B, n_head, T, head_dim)
            #        -> (B, n_head, T, head_dim)
            y = att @ v
        
        # 步骤4: 合并多头结果
        # 从 (B, n_head, T, head_dim) 转回 (B, T, n_head, head_dim)
        y = y.transpose(1, 2)
        # 连续化内存布局（view需要连续），然后合并为(B, T, C)
        y = y.contiguous().view(B, T, C)
        
        # 步骤5: 输出投影 + Dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    多层感知机（前馈网络）
    
    GPT-2使用GELU激活，扩展4倍维度再压缩回来
    结构: Linear -> GELU -> Linear
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 第一层: 扩展4倍（768 -> 3072）
        # 更多参数 = 更强的非线性表达能力
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # GELU激活: 平滑的ReLU变体，负数区域也有微小梯度
        self.gelu = nn.GELU()
        
        # 第二层: 压缩回原始维度（3072 -> 768）
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)   # 扩展
        x = self.gelu(x)   # 非线性变换
        x = self.c_proj(x) # 压缩
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer块（核心构建单元）
    
    结构（Pre-LN）:
        x = x + Attention(LN(x))   # 残差连接1
        x = x + MLP(LN(x))          # 残差连接2
    
    为什么叫"Pre-LN"？LayerNorm放在子层之前（而非之后）
    好处: 训练更稳定，梯度流动更顺畅
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # 第一个LayerNorm（Attention之前）
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 多头注意力
        self.attn = CausalSelfAttention(config)
        # 第二个LayerNorm（MLP之前）
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # 前馈网络
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接1: 先归一化，再Attention，再加回输入
        x = x + self.attn(self.ln_1(x))
        # 残差连接2: 先归一化，再MLP，再加回输入
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    完整的GPT模型
    
    组成:
    1. 词嵌入（wte）: token -> 向量
    2. 位置嵌入（wpe）: 位置 -> 向量  
    3. Dropout
    4. N个Transformer块（堆叠）
    5. 最终LayerNorm
    6. 语言模型头（lm_head）: 向量 -> token概率
    
    关键技巧: 权重绑定（weight tying）
    wte（输入嵌入）和lm_head（输出投影）共享权重矩阵
    减少参数量，提升泛化
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入: 50257个token，每个映射到768维
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # 位置嵌入: 1024个位置，每个映射到768维
        # 可学习的位置编码（区别于正弦位置编码）
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # Dropout（嵌入后）
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer块堆叠（n_layer个）
        # nn.ModuleList确保PyTorch能识别所有子模块
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # 最终LayerNorm
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # 语言模型头: 768维 -> 50257个token的概率
        # bias=False: 输出层不用偏置（跟随GPT-2设计）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 🔑 关键: 权重绑定
        # 输入嵌入和输出投影共享同一个权重矩阵
        # 形状都是(vocab_size, n_embd)，可以直接绑定
        self.wte.weight = self.lm_head.weight
        
        # 初始化所有权重
        self.apply(self._init_weights)
        
        # 特殊初始化: 残差投影的权重缩小
        # 根据GPT-2论文，每层残差路径的权重初始化标准差要除以sqrt(2*n_layer)
        # 防止深层网络方差爆炸
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, 
                    mean=0.0, 
                    std=0.02 / math.sqrt(2 * config.n_layer)
                )
    
    def _init_weights(self, module):
        """
        权重初始化策略
        
        Linear: 正态分布N(0, 0.02)
        Embedding: 同样N(0, 0.02)
        Bias: 初始化为0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            idx: 输入token IDs，形状 (batch_size, seq_len)
            targets: 目标token IDs（训练时用），相同形状
        
        返回:
            logits: 预测分数 (batch_size, seq_len, vocab_size)
            loss: 交叉熵损失（如果提供了targets）
        """
        device = idx.device
        b, t = idx.size()
        
        # 检查序列长度
        assert t <= self.config.block_size, (
            f"输入长度{t}超过最大长度{self.config.block_size}"
        )
        
        # 词嵌入: (b, t) -> (b, t, n_embd)
        tok_emb = self.wte(idx)
        
        # 位置嵌入: (t,) -> (t, n_embd)，然后广播到(b, t, n_embd)
        pos_emb = self.wpe(torch.arange(t, device=device))
        
        # 相加: 语义信息 + 位置信息
        x = self.drop(tok_emb + pos_emb)
        
        # 通过所有Transformer块
        for block in self.h:
            x = block(x)
        
        # 最终归一化
        x = self.ln_f(x)
        
        # 投影到词汇表维度，得到每个位置的分数
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        # 计算损失（如果提供了目标）
        loss = None
        if targets is not None:
            # 展平计算交叉熵:
            # logits: (b*t, vocab_size)
            # targets: (b*t,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100  # 忽略填充位置（如果有）
            )
        
        return logits, loss
    
    def crop_block_size(self, block_size: int):
        """
        裁剪模型支持的最大序列长度
        
        用途: 微调时如果新数据较短，可以裁剪以节省计算
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        
        # 裁剪位置嵌入
        self.wpe.weight = nn.Parameter(self.wpe.weight[:block_size])
        
        # 更新因果掩码（如果存在）
        for block in self.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
    
    @torch.no_grad()  # 禁用梯度，节省内存
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        自回归生成文本
        
        参数:
            idx: 初始token序列 (batch_size, seq_len)
            max_new_tokens: 要生成多少个新token
            temperature: 采样温度（<1更保守，>1更随机）
            top_k: 只从概率最高的k个token中采样（None=全部）
        
        返回:
            生成的完整序列 (batch_size, seq_len + max_new_tokens)
        """
        # 逐个token生成
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后block_size个
            # 这是模型的最大记忆长度
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 前向传播获取预测
            logits, _ = self.forward(idx_cond)
            
            # 只取最后一个位置的预测（预测下一个token）
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # 温度缩放: 除以temperature
            # T<1: 分布更尖锐，高概率token更突出
            # T>1: 分布更平坦，增加随机性
            logits = logits / temperature
            
            # Top-k筛选: 只保留概率最高的k个
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 低于第k名的都设为-inf（概率变为0）
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 计算概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 从多项分布中采样一个token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        计算模型参数量
        
        参数:
            non_embedding: 是否排除位置嵌入（通常与词嵌入共享）
        
        返回:
            参数总数
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 减去位置嵌入（通常不共享，不算关键参数）
            n_params -= self.wpe.weight.numel()
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        估计模型浮点运算利用率（Model Flops Utilization）
        
        与A100等GPU的理论峰值对比，评估训练效率
        
        参数:
            fwdbwd_per_iter: 每次迭代的forward+backward次数
            dt: 每次迭代的时间（秒）
        
        返回:
            MFU比例（0-1之间，越接近1越好）
        """
        # 估算每次forward+backward的浮点运算数
        # 基于PaLM论文的公式
        N = self.get_num_params()
        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_head,
            self.config.head_dim,
            self.config.block_size
        )
        
        # 每个token的flops
        flops_per_token = 6 * N + 12 * L * H * Q * T
        # 每次forward+backward（backward是forward的2倍）
        flops_per_fwdbwd = flops_per_token * T
        # 每次迭代
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # 实际达到的flops
        flops_achieved = flops_per_iter * (1.0 / dt)
        
        # A100 bfloat16峰值约312 TFLOPS
        flops_promised = 312e12
        
        mfu = flops_achieved / flops_promised
        return mfu


def create_model(config: ModelConfig) -> GPT:
    """
    工厂函数：根据配置创建模型
    
    用法:
        from nanogpt_plus.config import ModelConfig
        from nanogpt_plus.models import create_model
        
        cfg = ModelConfig(n_layer=12, n_embd=768)
        model = create_model(cfg)
    """
    return GPT(config)
