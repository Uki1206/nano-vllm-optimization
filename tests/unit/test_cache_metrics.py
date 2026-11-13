import unittest
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestCacheMetrics(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4
        self.params = SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True)

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def make_cached(self, blocks=4):
        manager = BlockManager(blocks, 4)
        seed = Sequence([1, 2, 3, 4, 5], self.params)
        manager.allocate(seed, manager.can_allocate(seed))
        seed.num_scheduled_tokens = 5
        manager.hash_blocks(seed)
        seed.num_cached_tokens = 5
        seed.num_scheduled_tokens = 0
        return manager, seed

    def test_hit_is_not_reuse_until_allocate(self):
        manager, seed = self.make_cached()
        request = Sequence([1, 2, 3, 4, 6], self.params)
        cached = manager.can_allocate(request)
        self.assertEqual(cached, 1)
        self.assertEqual(manager.metrics.hit_count, 1)
        self.assertEqual(manager.metrics.reused_tokens, 0)
        manager.allocate(request, cached)
        self.assertEqual(manager.metrics.reused_blocks, 1)
        self.assertEqual(manager.metrics.reused_tokens, 4)
        self.assertEqual(manager.blocks[seed.block_table[0]].ref_count, 2)
        manager.deallocate(seed)
        manager.deallocate(request)

    def test_free_cached_block_reused_and_snapshot_independent(self):
        manager, seed = self.make_cached()
        cached_block = seed.block_table[0]
        manager.deallocate(seed)
        before = manager.metrics.snapshot()
        request = Sequence([1, 2, 3, 4, 9], self.params)
        manager.allocate(request, manager.can_allocate(request))
        self.assertEqual(request.block_table[0], cached_block)
        self.assertEqual(manager.metrics.reused_tokens, 4)
        self.assertEqual(before["reused_tokens"], 0)

    def test_hit_then_allocation_failure_and_retry_counts(self):
        manager, seed = self.make_cached(blocks=2)
        request = Sequence([1, 2, 3, 4, 8], self.params)
        self.assertEqual(manager.can_allocate(request), -1)
        self.assertEqual(manager.can_allocate(request), -1)
        self.assertEqual(manager.metrics.hit_count, 2)
        self.assertEqual(manager.metrics.allocation_failure_count, 2)
        self.assertEqual(manager.metrics.reused_tokens, 0)
        manager.deallocate(seed)
        manager.allocate(request, manager.can_allocate(request))
        self.assertEqual(manager.metrics.hit_count, 3)
        self.assertEqual(manager.metrics.reused_tokens, 4)

    def test_miss_and_cached_mapping_eviction(self):
        manager, seed = self.make_cached(blocks=2)
        manager.deallocate(seed)
        missing = Sequence([9, 9, 9, 9, 9], self.params)
        self.assertEqual(manager.can_allocate(missing), 0)
        self.assertEqual(manager.metrics.hit_count, 0)
        manager.allocate(missing, 0)
        self.assertEqual(manager.metrics.eviction_count, 1)

    def test_scheduler_preemption_is_counted(self):
        config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=8, eos=-1,
                                 kvcache_block_size=4, num_kvcache_blocks=4)
        scheduler = Scheduler(config)
        request = Sequence([1, 2, 3], self.params)
        scheduler.add(request)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [10], mode)
        scheduler.preempt(request)
        self.assertEqual(scheduler.block_manager.metrics.preemption_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
