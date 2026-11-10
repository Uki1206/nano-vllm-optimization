import tempfile
import unittest
from types import SimpleNamespace

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_factory import build_scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.experimental.schedulers.unified_scheduler import UnifiedScheduler
from nanovllm.sampling_params import SamplingParams


def make_config(budget=5, max_seqs=4, blocks=16, backend="mixed"):
    return SimpleNamespace(
        max_num_seqs=max_seqs,
        max_num_batched_tokens=budget,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=blocks,
        scheduler_backend=backend,
    )


class TestUnifiedScheduler(unittest.TestCase):

    def setUp(self):
        self.previous_block_size = Sequence.block_size
        Sequence.block_size = 4
        self.params = SamplingParams(temperature=0.6, max_tokens=10, ignore_eos=True)

    def tearDown(self):
        Sequence.block_size = self.previous_block_size

    def test_factory_keeps_baseline_and_selects_mixed(self):
        self.assertIs(type(build_scheduler(make_config(backend="baseline"))), Scheduler)
        self.assertIsInstance(build_scheduler(make_config()), UnifiedScheduler)
        with self.assertRaises(ValueError):
            build_scheduler(make_config(backend="unknown"))

    def test_config_rejects_unsupported_mixed_modes(self):
        with tempfile.TemporaryDirectory() as model:
            with self.assertRaisesRegex(ValueError, "enforce_eager"):
                Config(model, scheduler_backend="mixed", enforce_eager=False)
            with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
                Config(model, scheduler_backend="mixed", enforce_eager=True, tensor_parallel_size=2)
            with self.assertRaisesRegex(ValueError, "Unknown scheduler backend"):
                Config(model, scheduler_backend="unknown")

    def test_decode_and_chunked_prefill_share_step(self):
        scheduler = UnifiedScheduler(make_config())
        request_a = Sequence([1, 2, 3], self.params)
        request_b = Sequence([4, 5, 6, 7, 8, 9, 10, 11], self.params)
        scheduler.add(request_a)

        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [request_a])
        scheduler.postprocess(seqs, [20], mode)

        scheduler.add(request_b)
        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [request_a, request_b])
        self.assertEqual([seq.num_scheduled_tokens for seq in seqs], [1, 4])
        self.assertEqual([seq.is_prefill for seq in seqs], [False, True])
        scheduler.postprocess(seqs, [21, 22], mode)
        self.assertEqual(request_a.num_completion_tokens, 2)
        self.assertEqual(request_b.num_cached_tokens, 4)
        self.assertEqual(request_b.num_completion_tokens, 0)

        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [request_a, request_b])
        self.assertEqual([seq.num_scheduled_tokens for seq in seqs], [1, 4])
        scheduler.postprocess(seqs, [23, 24], mode)
        self.assertEqual(request_b.num_completion_tokens, 1)
        self.assertEqual(request_b.status, SequenceStatus.RUNNING)

    def test_single_token_budget_alternates_without_starvation(self):
        scheduler = UnifiedScheduler(make_config(budget=1, max_seqs=1))
        request_a = Sequence([1], self.params)
        request_b = Sequence([2], self.params)
        scheduler.add(request_a)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [10], mode)

        scheduler.add(request_b)
        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [request_a])
        scheduler.postprocess(seqs, [11], mode)

        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [request_b])
        scheduler.postprocess(seqs, [12], mode)
        self.assertEqual(request_b.num_completion_tokens, 1)

    def test_waiting_requests_and_decode_rotate_fairly(self):
        scheduler = UnifiedScheduler(make_config(budget=2, max_seqs=2))
        requests = [Sequence([i], self.params) for i in range(1, 5)]
        for request in requests:
            scheduler.add(request)

        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, requests[:2])
        scheduler.postprocess(seqs, [10, 11], mode)

        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [requests[0], requests[2]])
        scheduler.postprocess(seqs, [12, 13], mode)

        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [requests[1], requests[3]])

    def test_preemption_keeps_block_lifecycle_consistent(self):
        scheduler = UnifiedScheduler(make_config(budget=8, blocks=2))
        params = SamplingParams(temperature=0.6, max_tokens=3, ignore_eos=True)
        request_a = Sequence([1, 2, 3], params)
        request_b = Sequence([4, 5, 6], params)
        scheduler.add(request_a)
        scheduler.add(request_b)

        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [10, 11], mode)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [12, 13], mode)

        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [request_a])
        self.assertEqual(request_b.status, SequenceStatus.WAITING)
        self.assertEqual(request_b.block_table, [])
        scheduler.postprocess(seqs, [14], mode)

        for _ in range(20):
            if scheduler.is_finished():
                break
            seqs, mode = scheduler.schedule()
            scheduler.postprocess(seqs, [15] * len(seqs), mode)

        self.assertTrue(scheduler.is_finished())
        self.assertEqual(request_a.num_completion_tokens, 3)
        self.assertEqual(request_b.num_completion_tokens, 3)
        self.assertEqual(len(scheduler.block_manager.used_block_ids), 0)
        self.assertTrue(all(block.ref_count == 0 for block in scheduler.block_manager.blocks))

    def test_no_capacity_reports_failure(self):
        scheduler = UnifiedScheduler(make_config(blocks=0))
        scheduler.add(Sequence([1], self.params))
        with self.assertRaisesRegex(RuntimeError, "KV cache capacity"):
            scheduler.schedule()


if __name__ == "__main__":
    unittest.main()
