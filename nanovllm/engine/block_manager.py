from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算链式哈希
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """
        内部方法：从free block池中拿一个物理block
        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        # 注意即使是从free池中取出的block,也不一定完全是"空"的，它的hash和token_ids可能残留还未被删除
        if(block.hash != -1 and self.hash_to_block_id[block.hash] == block_id):
            del self.hash_to_block_id[block.hash]  # 先删除对应的hash-block_id映射
        block.reset()  # 再清除block的hash,ref_count,token_ids
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """
        内部方法：把没人引用的block放回free block池
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        1.检查当前seq有多少prefix block可复用
        2.检查剩余kv block够不够分配
        """
        # --step1:初始化--
        h = -1
        num_cached_blocks = 0  # 可复用的block数(已经保存在kv cache中的块)
        num_new_blocks = seq.num_blocks   # 需要的新block数

        # --step2:遍历当前seq除最后一个block外的其他block，判断其是否能成为前缀，并更新num_cached_block和num_new_blocks的值--
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h,-1)
            # 如果当前h没有对应的block_id(-1)，或者h对应block的token_ids不是当前块的token_ids,说明当前块不是前缀
            # 因为理论上不同token_ids也可能算出相同的hash,因此需要hash和token_ids双重确认
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break   # 找到第一个不是前缀的块就可以退出了
            
            # 经过上面的双重判断没退出，则可确定当前块是前缀,但是前缀命中也有两种情况：
            # 1.前缀命中，且当前块处于used状态(还被其他请求使用)，因此不需要请求一个新block即可复用(num_new_block -= 1)
            # 2.前缀命中，但当前块处于free状态(曾经缓存过，但已经被释放)，这个块需要被重新分配回来(num_new_block不变)
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        # --step3:判断是否有足够空间分配--
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq:Sequence, num_cached_blocks: int):
        """
        真正给seq建立seq.block_table
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):    # 遍历前缀block
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            # 同样的两种情况，同上
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.append(block_id)
            seq.block_table.append(block_id)

        # 给除了前缀以外的块分配空间并加入block_table, 注意这里已经给一个seq提前预留了所有prefill的block
        for i in range(num_cached_blocks, seq.num_blocks):  
            seq.block_table.append(self._allocate_block())

        # 更新已有kv cache的token数，这里指有前缀的部分可以不用算
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """
        seq生成完成 或 被抢占 时，释放其占用的物理block(如果ref == 0)
        """
        for block_id in reversed(seq.block_table): # 这里是从后往前释放,让最有复用价值的前缀块沉到 free deque 队尾，最不易被覆写
            block = self.blocks[block_id]
            block.ref_count -= 1
            if(block.ref_count == 0):
                self._deallocate_block(block_id)
            seq.num_cached_tokens = 0
            seq.block_table.clear()

    def can_append(self, seq:Sequence) -> bool:
        """
        当前seq继续append token是否可行
        也就是说，如果这一步需要新的，free池中是否还有block
        (len(seq) % self.block_size = 1)就说明seq生成的token进入了一个新的逻辑block
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq:Sequence):
        """
        seq生成的token进入新的逻辑block时，为其分配物理空间
        """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq:Sequence):
        """
        把这轮刚计算完的（完整）block登记进prefix cache(hash_to_block_id)
        """
        # 先计算 要计算hash的完整blocks的起始位置和结束位置
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:  # 没产生新的block,不用hash
            return

        # 取start之前的链式哈希值
        h = self.blocks[seq.block_table[start-1]].hash if start > 0 else -1
        # 计算新block链式哈希
        for i in range(start, end):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id




