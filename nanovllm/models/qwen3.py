import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config


from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class Qwen3Attention(nn.Module):
    """
    Qwen3 的多头注意力模块。

    主要执行流程：
        Hidden States
            ↓
        QKV 投影
            ↓
        拆分 Q、K、V
            ↓
        调整形状为 [Token数, Head数, Head维度]
            ↓
        Q/K RMSNorm（取决于当前代码的 qkv_bias 设置）
            ↓
        RoPE 位置编码
            ↓
        Attention 计算
            ↓
        输出投影 O Projection
            ↓
        输出 Hidden States
    """

    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        num_kv_heads: int, 
        max_position: int = 4096*32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None, 
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()

        # ==================== 1. 计算注意力头数量 ====================
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # ==================== 2. QKV 投影 ====================
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # ==================== 3. Attention 输出投影 ====================
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False
        )

        # ==================== 4. RoPE 位置编码 ====================
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim = head_dim,
            max_position=max_position,
            base=rope_theta
        )
        # ==================== 5. Attention 计算模块 ====================
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        positions:
            Token 的位置索引，通常形状为 [num_tokens]。

        hidden_states:
            输入隐藏状态，形状为 [num_tokens, hidden_size]。

        返回：
            注意力输出，形状为 [num_tokens, hidden_size]。
        """

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 将线性投影结果转换成多头注意力需要的形状,-1表示自动推导
        q = q.view(-1, self.num_heads,self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 对 Q 和 K 执行 RMSNorm. V 不执行该归一化操作。
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 对 Q、K 应用 RoPE
        q, k = self.rotary_emb(positions, q, k)
        # 执行注意力计算
        o = self.attn(q, k, v)
        # 合并 Head 维度,[T, num_heads, head_dim] -> [T, num_heads * head_dim]
        output = self.o_proj(o.flatten(1, -1))  # flatten(1, -1) 保留第 0 维 Token 维度
        return output


"""
1.列并行：按输出维度切分权重，各 GPU 独立计算不同输出特征，
        适合输入完整、希望输出分片的场景。

2.行并行：按输入维度切分权重，各 GPU 计算同一个输出的部分和，
        需要 All-Reduce，适合输入已经分片、需要恢复完整输出的场景。

3.Transformer 中通常成对使用：QKV 和 MLP Up/Gate 使用列并行，
        O Projection 和 MLP Down 使用行并行，避免中间不必要的通信。

总结:列并行产生分片输出，行并行消费分片输入并通过求和生成完整输出。
"""


class Qwen3MLP(nn.Module):
    """
    Qwen3 的前馈网络（MLP）。

    主要流程：
        Hidden States
            ↓
        Gate / Up 投影
            ↓
        SiLU(Gate) * Up
            ↓
        Down 投影
            ↓
        输出 Hidden States
    """
     
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        # ==================== 1. Gate 和 Up 投影 ====================
        # 将 Gate Projection 和 Up Projection 合并为一个模块。
        # MergedColumnParallelLinear 对两个投影分别进行列并行切分。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False
        )
        # ==================== 2. Down 投影 ====================
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False
        )
        # 当前实现只支持 SiLU 激活
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    一个完整的 Qwen3 Transformer Decoder Layer。

    主要结构：

        Hidden States
             |
        Input LayerNorm
             |
        Self Attention
             |
        Residual + Post Attention LayerNorm
             |
            MLP
             |
        输出 Hidden States 和 Residual

    这里采用 Pre-Norm 结构，并将 residual 单独传递，
    以配合 RMSNorm 的融合实现，减少不必要的内存读写。
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()

        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # Attention 之前的归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Attention 之后、MLP 之前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states:
            当前层的主分支特征。

        residual:
            单独保存的残差分支；第一层输入时为 None。

        返回：
            hidden_states：当前 MLP 的输出。
            residual：累积到 Attention 输出为止的残差分支。
        """

        # 第一层没有上一层传来的独立 residual
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # 返回两个分支，供下一层继续执行融合残差计算
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 的 Transformer 主干网络。

    负责：
        Token Embedding
        -> 多层 Decoder Layer
        -> 最终 RMSNorm

    注意：这里不包含最终的 LM Head 投影。
    """
    def __init__(self, config: Qwen3Config,) -> None:
        super().__init__()
        # ==================== 1. Token Embedding ====================
        # 将 Token ID 转换为 hidden_size 维的向量。
        # 词表采用张量并行切分，每张 GPU 存储部分词表权重
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        # ==================== 2. Transformer 层堆叠 ====================
        # 创建 num_hidden_layers 个 Decoder Layer。
        # ModuleList 会注册其中的所有子模块和参数
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # ==================== 3. 最终归一化 ====================
        # 所有 Decoder Layer 执行完毕后，
        # 将最后一层的主分支和 residual 融合并进行归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        input_ids:
            输入 Token ID，通常为 [T]。

        positions:
            对应 Token 的位置索引，通常为 [T]。

        返回：
            最终隐藏状态，通常为 [T, hidden_size]。
        """
        # 1. Token ID -> Token Embedding
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        # 2. 依次执行所有 Decoder Layer
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 3. 最后一层结束后，执行最终 RMSNorm。
        # 将最后一层的 MLP 输出与 residual 融合
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型的顶层封装。

    包含：
        Qwen3Model：Transformer 主干
        ParallelLMHead：将隐藏状态映射到词表 logits

    这里将主干计算和 logits 计算拆开，
    方便推理引擎根据 Prefill/Decode 场景决定何时计算 logits。
    """

    # ==================== 权重名称映射 ====================

    # Hugging Face 权重通常将 Q/K/V、Gate/Up 保存为独立参数。
    # nano-vLLM 将它们融合为 QKV 和 Gate-Up 投影。
    # 该映射用于加载权重时，将原始参数名称对应到融合模块。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        # ==================== 1. Transformer 主干 ====================
        self.model = Qwen3Model(config)

        # ==================== 2. LM Head ====================
        # 将最后的隐藏状态映射到整个词表
        # 输入：[T, hidden_size]
        # 输出：[T, vocab_size]（完整 logits 通常由各 GPU 汇集）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # ==================== 3. 权重共享 ====================

        # 如果模型配置要求共享输入 Embedding 和输出 LM Head 的权重，
        # 则让两者指向同一份底层权重数据。
        # 这样可以避免为 Embedding 和 LM Head 分别保存一份权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ) -> torch.Tensor:
        # 执行 Transformer 主干，返回最终隐藏状态。
        # 注意：这里没有执行 LM Head，因此返回的不是 logits
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将最终隐藏状态投影到词表维度，得到 logits。
        # logits 表示每个位置上，各个 Token 的未归一化预测分数。
        # 后续可以通过 Softmax 和采样策略选择下一个 Token。
        return self.lm_head(hidden_states)


    
