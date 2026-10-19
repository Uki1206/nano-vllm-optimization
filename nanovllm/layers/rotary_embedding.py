from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin:torch.Tensor) -> torch.Tensor:
    """
    实现RoPE的核心计算:
    二维平面上有一个向量 (x1, x2)，将其逆时针旋转角度 theta 后，新向量 (y1, y2) 的计算公式:
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)) # 计算逆频率 theta_i（inv_freq）,形状为(head_size / 2,)
        t = torch.arange(max_position_embeddings, dtype=torch.float)        # t 是位置索引序列:[0, 1, 2, ..., L_max-1]，形状为 (max_position_embeddings,)。
        
        freqs = torch.einsum("i,j -> ij", t, inv_freq)          #计算向量 t 和 inv_freq 的外积
        # 结果 freqs 的形状为 (max_position_embeddings, head_size / 2)。
        # 其中第m行、第i列的元素就是位置m在第i组二维平面上的旋转角度 m*theta_i
        
        # 计算三角函数并构造缓存
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos,sin),dim=-1).unsqueeze_(1) #在第 1 维插入一个维度，形状变为 (max_position_embeddings, 1, head_size)。这个 1 是为了后续与包含多头维度 num_heads 的 query 和 key 相乘时能够自动广播
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]     #直接通过索引查表，取出当前这批 token 对应的三角函数值，形状为 (num_tokens, 1, head_size)
        cos, sin = cos_sin.chunk(2, dim=-1)         #将刚刚拼在一起的 cos 和 sin 拆开，各自形状恢复为 (num_tokens, 1, head_size / 2)
        
        # 对 Query 和 Key 执行旋转
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key
    """
    假设传入的 query 形状为 (num_tokens, num_heads, head_size)，
    key 形状为 (num_tokens, num_kv_heads, head_size)。

    调用上一段代码中的 apply_rotary_emb 函数后，
    cos 和 sin 中间的维度 1 会自动广播匹配 num_heads 和 num_kv_heads，
    对每个注意力头独立施加相同的旋转位置编码，最后返回编码后的 query 和 key。
    """