from nanovllm.engine.block_manager import Block, BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.experimental.cache.cache_metrics import CacheMetrics
from nanovllm.experimental.cache.eviction_policy import (
    EVICTION_POLICY_REGISTRY,
    EvictionPolicy,
)


class NoCacheBlockManager(BlockManager):
    """
    关闭 Prefix Cache 复用的对照组实现：不查表、不登记前缀哈希，仅管理物理块分配与回收。
    """

    def can_allocate(self, seq: Sequence) -> int:
        if len(self.free_block_ids) < seq.num_blocks:
            self.metrics.allocation_failure_count += 1
            return -1
        return 0

    def hash_blocks(self, seq: Sequence):
        return


class PolicyBlockManager(BlockManager):
    """
    基于可插拔 EvictionPolicy（FIFO / LRU）的 BlockManager。

    相比原始 BlockManager（单 deque）：
    1. 将未哈希/失效块与活跃缓存块分离，始终优先分配未哈希块，避免误淘汰有效前缀；
    2. 复用空闲缓存块为 O(1)，消除 deque.remove 的线性扫描；
    3. 严格维护 ref_count、hash_to_block_id、used_block_ids 与空闲池之间的一致性。
    """

    def __init__(self, num_blocks: int, block_size: int, policy: str | EvictionPolicy = "lru"):
        self.block_size = block_size
        self.metrics = CacheMetrics()
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.used_block_ids: set[int] = set()

        if isinstance(policy, EvictionPolicy):
            self.eviction_policy = policy
        else:
            policy_cls = EVICTION_POLICY_REGISTRY.get(policy)
            if policy_cls is None:
                raise ValueError(f"Unknown cache eviction policy: {policy}")
            self.eviction_policy = policy_cls(num_blocks)

        # 保持与基线一致的属性名（支持 len() 与 in 操作）
        self.free_block_ids = self.eviction_policy

    def _is_live_cached(self, block: Block) -> bool:
        return (
            block.hash != -1
            and self.hash_to_block_id.get(block.hash) == block.block_id
        )

    def _allocate_block(self) -> int:
        block_id = self.eviction_policy.pop_victim()
        block = self.blocks[block_id]
        if self._is_live_cached(block):
            del self.hash_to_block_id[block.hash]
            self.metrics.eviction_count += 1
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        self.eviction_policy.add(block_id, is_cached=self._is_live_cached(block))

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.eviction_policy.remove(block_id)
                self.used_block_ids.add(block_id)
            self.eviction_policy.on_cache_hit(block_id)
            seq.block_table.append(block_id)
            self.metrics.reused_blocks += 1
            self.metrics.reused_tokens += self.block_size

        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return

        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            prev_block_id = self.hash_to_block_id.get(h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
            self.eviction_policy.on_cache_insert(block_id)
            if prev_block_id is not None and prev_block_id != block_id:
                self.eviction_policy.demote_to_unhashed(prev_block_id)


def build_block_manager(config) -> BlockManager:
    policy = getattr(config, "cache_eviction_policy", "baseline")
    num_blocks = config.num_kvcache_blocks
    block_size = config.kvcache_block_size

    if policy == "baseline":
        return BlockManager(num_blocks, block_size)
    if policy == "none":
        return NoCacheBlockManager(num_blocks, block_size)
    if policy in EVICTION_POLICY_REGISTRY:
        return PolicyBlockManager(num_blocks, block_size, policy=policy)
    raise ValueError(f"Unknown cache_eviction_policy: {policy}")
