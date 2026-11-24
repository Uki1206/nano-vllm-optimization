import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.experimental.cache.block_manager_lru import (
    NoCacheBlockManager,
    PolicyBlockManager,
    build_block_manager,
)
from nanovllm.sampling_params import SamplingParams


def assert_block_manager_invariants(test_case: unittest.TestCase, manager: BlockManager, num_blocks: int):
    """校验 BlockManager 核心生命周期不变量。"""
    used = manager.used_block_ids
    free = manager.free_block_ids
    test_case.assertEqual(len(used) + len(free), num_blocks)

    for block_id in range(num_blocks):
        block = manager.blocks[block_id]
        in_used = block_id in used
        in_free = block_id in free
        test_case.assertNotEqual(in_used, in_free, f"block {block_id} must be exclusively used or free")
        if in_used:
            test_case.assertGreaterEqual(block.ref_count, 1)
        else:
            test_case.assertEqual(block.ref_count, 0)

    for h, block_id in manager.hash_to_block_id.items():
        test_case.assertEqual(manager.blocks[block_id].hash, h)


class TestBlockManagerLifecycle(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4
        self.params = SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True)

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def make_manager(self, policy: str, num_blocks: int = 8, block_size: int = 4) -> BlockManager:
        cfg = SimpleNamespace(
            num_kvcache_blocks=num_blocks,
            kvcache_block_size=block_size,
            cache_eviction_policy=policy,
        )
        return build_block_manager(cfg)

    def test_factory_builds_expected_types_and_validates(self):
        self.assertIs(type(self.make_manager("baseline")), BlockManager)
        self.assertIsInstance(self.make_manager("none"), NoCacheBlockManager)
        self.assertIsInstance(self.make_manager("fifo"), PolicyBlockManager)
        self.assertIsInstance(self.make_manager("lru"), PolicyBlockManager)

        with self.assertRaises(ValueError):
            self.make_manager("unknown_policy")

    def test_allocate_append_deallocate_invariants_all_policies(self):
        for policy in ("baseline", "none", "fifo", "lru"):
            with self.subTest(policy=policy):
                num_blocks = 6
                manager = self.make_manager(policy, num_blocks=num_blocks)
                assert_block_manager_invariants(self, manager, num_blocks)

                seq = Sequence([1, 2, 3, 4], self.params)  # 刚好 1 个完整块
                cached = manager.can_allocate(seq)
                self.assertEqual(cached, 0)
                manager.allocate(seq, cached)
                seq.num_scheduled_tokens = 4
                manager.hash_blocks(seq)
                seq.num_cached_tokens = 4
                seq.num_scheduled_tokens = 0
                assert_block_manager_invariants(self, manager, num_blocks)

                # Decode 跨过块边界：第 5 个 token 需要分配新块
                seq.append_token(5)
                self.assertTrue(manager.can_append(seq))
                manager.may_append(seq)
                self.assertEqual(len(seq.block_table), 2)
                assert_block_manager_invariants(self, manager, num_blocks)

                manager.deallocate(seq)
                self.assertEqual(seq.block_table, [])
                self.assertEqual(seq.num_cached_tokens, 0)
                assert_block_manager_invariants(self, manager, num_blocks)

    def test_hash_collision_rejected_by_token_content_check(self):
        for policy in ("baseline", "fifo", "lru"):
            with self.subTest(policy=policy):
                manager = self.make_manager(policy, num_blocks=6)
                with patch.object(BlockManager, "compute_hash", return_value=123456789):
                    seq1 = Sequence([1, 2, 3, 4, 5], self.params)
                    manager.allocate(seq1, manager.can_allocate(seq1))
                    seq1.num_scheduled_tokens = 5
                    manager.hash_blocks(seq1)
                    seq1.num_cached_tokens = 5
                    seq1.num_scheduled_tokens = 0

                    # 不同 token 内容但强制相同哈希值，必须拒绝错误复用
                    seq2 = Sequence([9, 9, 9, 9, 5], self.params)
                    cached = manager.can_allocate(seq2)
                    self.assertEqual(cached, 0)
                    self.assertEqual(manager.metrics.hit_count, 0)
                    manager.allocate(seq2, cached)
                    self.assertNotEqual(seq2.block_table[0], seq1.block_table[0])
                    assert_block_manager_invariants(self, manager, 6)

    def test_referenced_blocks_are_never_evicted(self):
        for policy in ("fifo", "lru"):
            with self.subTest(policy=policy):
                manager = self.make_manager(policy, num_blocks=2)
                seq1 = Sequence([1, 2, 3, 4, 5], self.params)  # 占用 2 块（满）
                manager.allocate(seq1, manager.can_allocate(seq1))
                seq1.num_scheduled_tokens = 5
                manager.hash_blocks(seq1)
                seq1.num_cached_tokens = 5
                seq1.num_scheduled_tokens = 0

                # 所有块 ref_count == 1，新请求无法分配，不得淘汰正在被引用的块
                seq2 = Sequence([10, 11, 12, 13, 14], self.params)
                self.assertEqual(manager.can_allocate(seq2), -1)
                self.assertEqual(manager.metrics.eviction_count, 0)
                self.assertEqual(manager.blocks[seq1.block_table[0]].ref_count, 1)
                assert_block_manager_invariants(self, manager, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
