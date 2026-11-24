from abc import ABC, abstractmethod
from collections import OrderedDict
import heapq


class EvictionPolicy(ABC):
    """
    管理 ref_count == 0 的空闲物理块池，决定下一次分配时回收哪一个 Block。

    所有具体策略都严格遵守两级优先级：
    1. 未哈希块（hash == -1）或已被覆盖失效的陈旧块（shadowed）优先回收；
    2. 只有无可优先回收块时，才按策略（FIFO / LRU）淘汰活跃缓存块。
    """

    def __init__(self, num_blocks: int):
        # 初始所有物理块都未写入有效前缀哈希，全部放入优先回收池
        self._unhashed: OrderedDict[int, None] = OrderedDict(
            (block_id, None) for block_id in range(num_blocks)
        )

    @property
    def num_unhashed(self) -> int:
        return len(self._unhashed)

    @property
    @abstractmethod
    def num_cached(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._unhashed) + self.num_cached

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def add(self, block_id: int, is_cached: bool = False) -> None:
        """将一个 ref_count 归零的块加入空闲池。"""
        raise NotImplementedError

    @abstractmethod
    def remove(self, block_id: int) -> None:
        """将一个空闲块移出空闲池（例如命中缓存后重新挂载到 Sequence）。"""
        raise NotImplementedError

    @abstractmethod
    def pop_victim(self) -> int:
        """弹出一个待分配的物理块（优先未哈希块，否则按策略淘汰缓存块）。"""
        raise NotImplementedError

    def on_cache_insert(self, block_id: int) -> None:
        """当块写入新的有效 Prefix Hash 时触发。"""

    def on_cache_hit(self, block_id: int) -> None:
        """当块作为 Prefix Cache 被复用时触发。"""

    @abstractmethod
    def demote_to_unhashed(self, block_id: int) -> None:
        """当某个空闲缓存块的 Hash 映射被新块覆盖失效时，降级到优先回收池。"""
        raise NotImplementedError


class FIFOEvictionPolicy(EvictionPolicy):
    """
    按缓存写入顺序（First-In-First-Out）淘汰活跃缓存块。

    即使热点前缀被多次复用，其原始写入序号不变；缓存承压时仍会优先淘汰最早写入的块。
    使用最小堆 + 惰性删除维护缓存块的写入顺序，remove 为 O(1)，add/pop 为均摊 O(log N)。
    """

    def __init__(self, num_blocks: int):
        super().__init__(num_blocks)
        self._cached: set[int] = set()
        self._insert_order: dict[int, int] = {}
        self._heap: list[tuple[int, int]] = []
        self._clock = 0

    @property
    def num_cached(self) -> int:
        return len(self._cached)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._unhashed or block_id in self._cached

    def on_cache_insert(self, block_id: int) -> None:
        self._clock += 1
        self._insert_order[block_id] = self._clock

    def add(self, block_id: int, is_cached: bool = False) -> None:
        if block_id in self._unhashed or block_id in self._cached:
            raise ValueError(f"Block {block_id} is already in free pool")
        if not is_cached:
            self._unhashed[block_id] = None
            return
        order = self._insert_order.get(block_id)
        if order is None:
            self._clock += 1
            order = self._clock
            self._insert_order[block_id] = order
        self._cached.add(block_id)
        heapq.heappush(self._heap, (order, block_id))

    def remove(self, block_id: int) -> None:
        if block_id in self._unhashed:
            del self._unhashed[block_id]
        elif block_id in self._cached:
            self._cached.remove(block_id)
        else:
            raise KeyError(f"Block {block_id} not in free pool")

    def pop_victim(self) -> int:
        if self._unhashed:
            block_id, _ = self._unhashed.popitem(last=False)
            return block_id

        while self._heap:
            order, block_id = heapq.heappop(self._heap)
            if block_id in self._cached and self._insert_order.get(block_id) == order:
                self._cached.remove(block_id)
                return block_id

        raise RuntimeError("No free block available for eviction")

    def demote_to_unhashed(self, block_id: int) -> None:
        if block_id in self._cached:
            self._cached.remove(block_id)
            self._unhashed[block_id] = None


class LRUEvictionPolicy(EvictionPolicy):
    """
    按最近最少使用（Least Recently Used）淘汰活跃缓存块。

    使用两个 OrderedDict 分别维护优先回收池和活跃缓存池：
    - 复用空闲缓存块（remove）为 O(1)，消除原 deque.remove 的 O(N) 扫描；
    - 请求释放时按反序归还，使同一序列中更深的后缀块比浅层前缀块先被淘汰；
    - 热点前缀被复用后重新释放会回到 MRU 端，避免被冷门请求冲刷掉。
    """

    def __init__(self, num_blocks: int):
        super().__init__(num_blocks)
        self._cached: OrderedDict[int, None] = OrderedDict()

    @property
    def num_cached(self) -> int:
        return len(self._cached)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._unhashed or block_id in self._cached

    def add(self, block_id: int, is_cached: bool = False) -> None:
        if block_id in self._unhashed or block_id in self._cached:
            raise ValueError(f"Block {block_id} is already in free pool")
        if is_cached:
            # 放入 MRU 端（队尾）
            self._cached[block_id] = None
        else:
            self._unhashed[block_id] = None

    def remove(self, block_id: int) -> None:
        if block_id in self._unhashed:
            del self._unhashed[block_id]
        elif block_id in self._cached:
            del self._cached[block_id]
        else:
            raise KeyError(f"Block {block_id} not in free pool")

    def pop_victim(self) -> int:
        if self._unhashed:
            block_id, _ = self._unhashed.popitem(last=False)
            return block_id
        if self._cached:
            # 弹出 LRU 端（队头）
            block_id, _ = self._cached.popitem(last=False)
            return block_id
        raise RuntimeError("No free block available for eviction")

    def demote_to_unhashed(self, block_id: int) -> None:
        if block_id in self._cached:
            del self._cached[block_id]
            self._unhashed[block_id] = None


EVICTION_POLICY_REGISTRY = {
    "fifo": FIFOEvictionPolicy,
    "lru": LRUEvictionPolicy,
}
