import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler_factory import build_scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.experimental.schedulers.budget_controller import PrefillBudgetController
from nanovllm.experimental.schedulers.budgeted_scheduler import BudgetedScheduler
from nanovllm.sampling_params import SamplingParams


def make_config(mode="adaptive", budget=8):
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=budget,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=16,
        scheduler_backend="budgeted",
        prefill_budget_mode=mode,
        min_prefill_budget=2,
        max_prefill_budget=budget,
        initial_prefill_budget=4,
        target_itl_ms=50.0,
        itl_ema_alpha=1.0,
    )


class TestPrefillBudget(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4
        self.params = SamplingParams(temperature=0.6, max_tokens=10, ignore_eos=True)

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_controller_smooths_clamps_and_holds_without_waiters(self):
        controller = PrefillBudgetController(2, 8, 8, 50, 0.5)
        self.assertEqual(controller.observe_itl(70, True), 6)
        self.assertEqual(controller.observe_itl(10, True), 6)  # EMA=40，位于缓冲区
        self.assertAlmostEqual(controller.ema_itl_ms, 40)
        self.assertEqual(controller.observe_itl(10, True), 8)
        self.assertEqual(controller.observe_itl(1000, False), 8)
        for _ in range(20):
            controller.observe_itl(1000, True)
        self.assertEqual(controller.current_budget, 2)

    def test_total_prefill_cap_and_decode_share_global_budget(self):
        scheduler = build_scheduler(make_config())
        self.assertIsInstance(scheduler, BudgetedScheduler)
        a = Sequence([1], self.params)
        scheduler.add(a)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [20], mode)

        b = Sequence([2] * 8, self.params)
        c = Sequence([3] * 2, self.params)
        scheduler.add(b)
        scheduler.add(c)
        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [a, b])
        self.assertEqual([s.num_scheduled_tokens for s in seqs], [1, 4])
        self.assertLessEqual(sum(s.num_scheduled_tokens for s in seqs), 8)
        scheduler.postprocess(seqs, [21, 22], mode)
        self.assertEqual(b.num_cached_tokens, 4)
        self.assertEqual(b.num_completion_tokens, 0)

    def test_prefill_cap_is_shared_by_multiple_requests(self):
        scheduler = BudgetedScheduler(make_config())
        a = Sequence([1], self.params)
        scheduler.add(a)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [20], mode)
        b = Sequence([2] * 2, self.params)
        c = Sequence([3] * 3, self.params)
        scheduler.add(b)
        scheduler.add(c)
        seqs, mode = scheduler.schedule()
        self.assertEqual(seqs, [a, b, c])
        self.assertEqual([s.num_scheduled_tokens for s in seqs], [1, 2, 2])
        self.assertEqual(sum(s.num_scheduled_tokens for s in seqs if s.is_prefill), 4)

    def test_measured_itl_changes_next_budget_only_when_waiting(self):
        scheduler = BudgetedScheduler(make_config())
        a = Sequence([1], self.params)
        scheduler.add(a)
        seqs, mode = scheduler.schedule()
        before = [s.num_completion_tokens for s in seqs]
        scheduler.postprocess(seqs, [20], mode)
        scheduler.observe_step(seqs, before, 1.0, False)

        seqs, mode = scheduler.schedule()
        before = [s.num_completion_tokens for s in seqs]
        scheduler.postprocess(seqs, [21], mode)
        scheduler.observe_step(seqs, before, 1.1, True)
        self.assertAlmostEqual(scheduler.controller.ema_itl_ms, 100)
        self.assertEqual(scheduler.controller.current_budget, 3)
        self.assertEqual(scheduler.last_prefill_budget, 4)
        scheduler.schedule()
        self.assertEqual(scheduler.last_prefill_budget, 3)

    def test_fixed_budget_never_changes(self):
        scheduler = BudgetedScheduler(make_config(mode="fixed"))
        a = Sequence([1], self.params)
        scheduler.add(a)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [20], mode)
        scheduler.observe_step(seqs, [0], 1.0, False)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [21], mode)
        scheduler.observe_step(seqs, [1], 1.1, True)
        self.assertEqual(scheduler.controller.current_budget, 4)

    def test_finished_requests_remove_timestamps(self):
        scheduler = BudgetedScheduler(make_config())
        params = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
        a = Sequence([1], params)
        scheduler.add(a)
        seqs, mode = scheduler.schedule()
        scheduler.postprocess(seqs, [20], mode)
        scheduler.observe_step(seqs, [0], 1.0, False)
        self.assertNotIn(a.seq_id, scheduler.last_token_times)

    def test_invalid_bounds_rejected(self):
        config = make_config()
        config.min_prefill_budget = 9
        with self.assertRaises(ValueError):
            BudgetedScheduler(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
