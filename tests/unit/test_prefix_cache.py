import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.experimental.cache.block_manager_lru import build_block_manager
from nanovllm.sampling_params import SamplingParams
from tests.unit.test_block_manager import assert_block_manager_invariants


class TestPrefixCacheEviction(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4
        self.params = SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True)

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def make_manager(self, policy: str, num_blocks: int, block_size: int = 4):
        cfg = SimpleNamespace(
            num_kvcache_blocks=num_blocks,
            kvcache_block_size=block_size,
            cache_eviction_policy=policy,
        )
        return build_block_manager(cfg)

    def run_single_request(self, manager, token_ids: list[int]) -> int:
        """模拟单个请求完整 Prefill + 注册 Prefix Hash + 释放，返回复用块数。"""
        seq = Sequence(token_ids, self.params)
        cached = manager.can_allocate(seq)
        self.assertGreaterEqual(cached, 0, "expected enough free blocks")
        manager.allocate(seq, cached)
        seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
        manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        manager.deallocate(seq)
        return cached

    def test_unhashed_tail_blocks_reclaimed_before_cached_prefixes(self):
        """
        证明两级空闲池的价值：
        当短请求的未哈希尾块（hash == -1）释放回池中时，baseline 单 deque 会把无用尾块排在
        缓存前缀块后面，导致下一次分配误淘汰有效前缀；而 LRU/FIFO 始终先回收未哈希块。
        """
        num_blocks = 4
        seed_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 2 个完整缓存块 + 1 个未满尾块
        short_tokens = [99, 98]                    # 1 个未满尾块（无缓存价值）
        new_tokens = [20, 21, 22, 23, 24]          # 需要 2 个新块
        probe_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 10]

        # Baseline 复现提前误淘汰
        bm_base = self.make_manager("baseline", num_blocks=num_blocks)
        self.run_single_request(bm_base, seed_tokens)
        self.run_single_request(bm_base, short_tokens)
        self.run_single_request(bm_base, new_tokens)
        self.assertEqual(bm_base.metrics.eviction_count, 1)
        self.assertEqual(self.run_single_request(bm_base, probe_tokens), 1)

        # LRU 与 FIFO 优先回收两个未哈希块，零误淘汰，完整保留 2 个缓存前缀块
        for policy in ("fifo", "lru"):
            with self.subTest(policy=policy):
                bm = self.make_manager(policy, num_blocks=num_blocks)
                self.run_single_request(bm, seed_tokens)
                self.run_single_request(bm, short_tokens)
                self.run_single_request(bm, new_tokens)
                self.assertEqual(bm.metrics.eviction_count, 0)
                self.assertEqual(self.run_single_request(bm, probe_tokens), 2)
                assert_block_manager_invariants(self, bm, num_blocks)

    def test_lru_retains_hot_prefix_where_fifo_evicts_it(self):
        """
        经典热点 vs 冷扫描场景（容量仅够放 2 个完整前缀 + 1 个工作尾块，总计 3 块）：
        - 先写入热点前缀 H，再写入冷前缀 C1；
        - 再次命中热点 H（刷新 recency，但不改变 FIFO 写入顺序）；
        - 写入新冷前缀 C2，触发 1 次缓存淘汰：
          * FIFO 淘汰最早写入的 H -> 后续 H 请求 Miss；
          * LRU 淘汰最久未使用的 C1 -> 后续 H 请求 Hit！
        """
        num_blocks = 3
        hot_1 = [1, 2, 3, 4, 100]
        hot_2 = [1, 2, 3, 4, 101]
        hot_3 = [1, 2, 3, 4, 102]
        cold_1 = [11, 12, 13, 14, 200]
        cold_2 = [21, 22, 23, 24, 300]

        # FIFO：命中 H 不更新其写入顺序，C2 淘汰 H
        bm_fifo = self.make_manager("fifo", num_blocks=num_blocks)
        self.assertEqual(self.run_single_request(bm_fifo, hot_1), 0)
        self.assertEqual(self.run_single_request(bm_fifo, cold_1), 0)
        self.assertEqual(self.run_single_request(bm_fifo, hot_2), 1)  # 命中 H
        self.assertEqual(self.run_single_request(bm_fifo, cold_2), 0) # 淘汰 H
        self.assertEqual(self.run_single_request(bm_fifo, hot_3), 0)  # H 已被淘汰 -> Miss
        assert_block_manager_invariants(self, bm_fifo, num_blocks)

        # LRU：命中 H 后 H 回到 MRU，C2 淘汰冷块 C1，H 保持命中
        bm_lru = self.make_manager("lru", num_blocks=num_blocks)
        self.assertEqual(self.run_single_request(bm_lru, hot_1), 0)
        self.assertEqual(self.run_single_request(bm_lru, cold_1), 0)
        self.assertEqual(self.run_single_request(bm_lru, hot_2), 1)   # 命中并刷新 H
        self.assertEqual(self.run_single_request(bm_lru, cold_2), 0)  # 淘汰冷块 C1
        self.assertEqual(self.run_single_request(bm_lru, hot_3), 1)   # H 依然命中！
        self.assertEqual(bm_lru.metrics.reused_blocks, 2)
        assert_block_manager_invariants(self, bm_lru, num_blocks)

    def test_lru_evicts_deeper_suffix_before_root_prefix(self):
        """
        同一长序列释放时按倒序归还，使深层后缀块比浅层根前缀块先被 LRU 淘汰。
        """
        num_blocks = 4
        bm = self.make_manager("lru", num_blocks=num_blocks)
        # 3 个完整前缀块 (B0, B1, B2) + 1 个尾块 (B3)
        long_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.assertEqual(self.run_single_request(bm, long_prompt), 0)

        # 新请求需要 2 个块：先回收 B3（未哈希），再淘汰 1 个缓存块，应淘汰最深的 B2
        other = [90, 91, 92, 93, 94]
        self.assertEqual(self.run_single_request(bm, other), 0)
        self.assertEqual(bm.metrics.eviction_count, 1)

        # 再次查询 long_prompt，浅层的 B0 和 B1 仍完整保留（复用 2 块）
        probe = [1, 2, 3, 4, 5, 6, 7, 8, 99]
        self.assertEqual(self.run_single_request(bm, probe), 2)
        assert_block_manager_invariants(self, bm, num_blocks)

    def test_shadowed_duplicate_hash_demoted_to_unhashed(self):
        """
        两个并发请求写入相同前缀哈希时，被覆盖的旧物理块失效（shadowed）；
        后续淘汰必须优先回收陈旧块，不能浪费活跃缓存槽位。
        """
        num_blocks = 5
        bm = self.make_manager("lru", num_blocks=num_blocks)

        seq_a = Sequence([1, 2, 3, 4, 5], self.params)
        seq_b = Sequence([1, 2, 3, 4, 6], self.params)
        # 并发分配，尚未 hash，各自拿到独立物理块
        bm.allocate(seq_a, bm.can_allocate(seq_a))
        bm.allocate(seq_b, bm.can_allocate(seq_b))
        self.assertNotEqual(seq_a.block_table[0], seq_b.block_table[0])

        # seq_a 先完成并释放，其块进入 cached 池
        seq_a.num_scheduled_tokens = 5
        bm.hash_blocks(seq_a)
        seq_a.num_cached_tokens = 5
        seq_a.num_scheduled_tokens = 0
        bm.deallocate(seq_a)

        # seq_b 后完成，覆盖 hash_to_block_id[h] -> seq_a 的块应立即降级到 unhashed 池
        seq_b.num_scheduled_tokens = 5
        bm.hash_blocks(seq_b)
        seq_b.num_cached_tokens = 5
        seq_b.num_scheduled_tokens = 0
        bm.deallocate(seq_b)

        # 此时全池 5 块中，仅有 1 个活跃缓存块（seq_b[0]），其余 4 块都应处于 unhashed 优先回收池
        self.assertEqual(bm.eviction_policy.num_cached, 1)
        self.assertEqual(bm.eviction_policy.num_unhashed, 4)
        assert_block_manager_invariants(self, bm, num_blocks)

    def test_eviction_removes_stale_hash_and_no_cache_disables_reuse(self):
        bm_none = self.make_manager("none", num_blocks=4)
        self.assertEqual(self.run_single_request(bm_none, [1, 2, 3, 4, 5]), 0)
        self.assertEqual(self.run_single_request(bm_none, [1, 2, 3, 4, 6]), 0)
        self.assertEqual(bm_none.metrics.lookup_count, 0)
        self.assertEqual(bm_none.metrics.reused_tokens, 0)
        self.assertEqual(len(bm_none.hash_to_block_id), 0)

        # 验证淘汰后旧哈希映射已彻底删除，不会残留脏映射
        bm_lru = self.make_manager("lru", num_blocks=2)
        self.run_single_request(bm_lru, [1, 2, 3, 4, 5])
        self.assertEqual(len(bm_lru.hash_to_block_id), 1)
        old_hash = next(iter(bm_lru.hash_to_block_id.keys()))
        self.run_single_request(bm_lru, [9, 9, 9, 9, 9])
        self.assertNotIn(old_hash, bm_lru.hash_to_block_id)
        self.assertEqual(bm_lru.metrics.eviction_count, 1)
        assert_block_manager_invariants(self, bm_lru, 2)

    def test_scheduler_preemption_and_recovery_preserves_invariants(self):
        """在紧张显存下配合 Scheduler 跑完抢占与重算恢复，每步均保持不变量成立。"""
        for policy in ("baseline", "none", "fifo", "lru"):
            with self.subTest(policy=policy):
                num_blocks = 5
                cfg = SimpleNamespace(
                    max_num_seqs=4,
                    max_num_batched_tokens=16,
                    eos=-1,
                    kvcache_block_size=4,
                    num_kvcache_blocks=num_blocks,
                    cache_eviction_policy=policy,
                )
                scheduler = Scheduler(cfg)
                sp = SamplingParams(temperature=0.6, max_tokens=6, ignore_eos=True)
                scheduler.add(Sequence([1, 2, 3, 4, 5], sp))
                scheduler.add(Sequence([1, 2, 3, 4, 6], sp))
                scheduler.add(Sequence([7, 8, 9, 10, 11], sp))

                steps = 0
                while not scheduler.is_finished():
                    seqs, is_prefill = scheduler.schedule()
                    assert_block_manager_invariants(self, scheduler.block_manager, num_blocks)
                    scheduler.postprocess(seqs, [42 + steps] * len(seqs), is_prefill)
                    assert_block_manager_invariants(self, scheduler.block_manager, num_blocks)
                    steps += 1
                    self.assertLess(steps, 100, "scheduler should make progress and finish")

                self.assertEqual(len(scheduler.block_manager.used_block_ids), 0)
                self.assertEqual(len(scheduler.block_manager.free_block_ids), num_blocks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
