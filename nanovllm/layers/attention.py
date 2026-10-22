import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


"""
为什么cache不需要传 k_cache_stride？
因为 k_cache 是专门独立分配的一整块连续显存
（形状通常为 [num_blocks, block_size, num_kv_heads, head_dim]）。
外部函数的 assert k_cache.stride(1) == D 已经保证了它在每个槽位（Slot）内部和槽位之间
都是紧密排列、没有夹杂其他数据的。所以在核函数里直接用 slot * D 就能算出偏移量，的步长固定就是 D

v也同理


key_stride为什么不能直接用 D（即 num_kv_heads * head_dim）？
在大模型中，通常会用一个大的线性层 qkv_proj 一次性算出拼接在一起的 [Q, K, V]，
然后再通过切片（Slice/Split）分离出 q, k, v。
切片后的 k 在内存中并不是独立连续的——第 0 个 Token 的 k 和第 1 个 Token 的 k 之间，还夹着 q 和 v 的数据。
因此，必须传入真实的 key.stride(0)，通过 idx * key_stride 才能准确跳到第 idx 个 Token 的 key 所在位置。
"""

@triton.jit     #装饰器，将 Python 函数编译为在 GPU 上运行的机器码。
def store_kvcache_kernel(
    key_ptr,        # 当前步计算出的 key 张量在显存中的起始指针,[N, num_kv_heads, head_dim]
    key_stride,     # 在 key 张量中，第 0 维（即 Token 维度）每跨过 1 个 Token，在物理内存中需要跳过的元素个数
    value_ptr,
    value_stride,   # 同上
    k_cache_ptr,    # 预先分配好的全局 Key 缓存池在显存中的起始指针
    v_cache_ptr,    # 预先分配好的全局 Value 缓存池在显存中的起始指针
    slot_mapping_ptr,   # 槽位映射表张量在显存中的起始指针
    D: tl.constexpr   #编译期常量，表示一个 Token 的所有 K Head 拼接后的维度，同时也是所有 V Head 拼接后的维度（即 num_kv_heads * head_dim）
):
    """
    并发地把当前步算出来的每个Token的key和value，按照slot_mapping指引的位置，精准写入全局KV Cache显存池中
    """
    idx = tl.program_id(0)                  # 获取当前线程块（Program）的 ID，第 idx 个线程块专门负责搬运第 idx 个 Token 的数据
    slot = tl.load(slot_mapping_ptr + idx)  # 读取第 idx 个 Token 应该存入 KV Cache 的哪个物理槽位（Slot）
    if slot == -1:
        return
    
    key_offsets = idx * key_stride + tl.arange(0, D)    # 定位到当前第 idx 个 Token 在输入张量 key 中的连续内存地址（长度为 D）
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)      # 将该 Token 的 key 和 value 向量从全局显存加载到 GPU 寄存器（SRAM）中
    
    cache_offsets = slot * D + tl.arange(0 ,D)      # 根据查到的物理槽位 slot，计算出在 k_cache 和 v_cache 中的目标内存地址
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)    # 将寄存器里的 key 和 value 写入 KV Cache 对应的槽位中

    # 理解：一个token对应一个cache中的slot位置，D是两个token的k或v在cache中的间隔，因此slot从0开始的话，slot*D就是先算出当前token k_cache的首地址
    # 但是在原来的key,value张量中，两个token的k/v之间的间隔（元素个数）不等于D（QKV拼在一个大张量里了），因此要拿到实际的间隔key_stride

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    负责检查输入 Tensor 的形状和内存布局，然后启动 Triton Kernel，将 Key、Value 写入 KV Cache
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 1.检查 Key 和 Value 的最后一维是否连续 -> 保证同一个 Head 内部的元素在显存中连续存储
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    # 2.检查相邻 Head 之间的内存跨度 -> 保证同一个 Token 的不同 Head 在显存中紧密排列
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # 3.检查 KV Cache 中相邻 Token Slot 的内存跨度 -> 每个 Slot 存储 D 个 Key 元素或 D 个 Value 元素
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    # 4.确保每个 Token 都有对应的物理 Slot
    assert slot_mapping.numel() == N
    # 5.启动 Triton Kernel
    # grid=(N,)：启动 N 个 Triton Program，每个 Program 负责一个 Token
    # key.stride(0)：相邻 Token 的 Key 起始位置之间的元素跨度
    # value.stride(0)：相邻 Token 的 Value 起始位置之间的元素跨度
    # D：每个 Token 的 Key 或 Value 向量长度
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o