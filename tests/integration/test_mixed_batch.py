import atexit
import os
import unittest

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.experimental.schedulers.budgeted_scheduler import BudgetedScheduler
from nanovllm.experimental.schedulers.unified_scheduler import UnifiedScheduler


MODEL_PATH = os.environ.get("NANOVLLM_TEST_MODEL")


class GreedyRecorder:
    # 验证时使用贪心选择，避免随机采样掩盖输出差异
    def __init__(self):
        self.seqs = []
        self.logits = {}

    def __call__(self, logits, temperatures):
        rows = logits.detach().float().cpu()
        for seq, row in zip(self.seqs, rows):
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.logits.setdefault(seq.seq_id, []).append(row)
        return logits.argmax(dim=-1)


@unittest.skipUnless(MODEL_PATH and torch.cuda.is_available(), "需要设置 NANOVLLM_TEST_MODEL 和 CUDA")
class TestMixedBatchGPU(unittest.TestCase):

    def test_mixed_matches_isolated_requests(self):
        llm = LLM(
            MODEL_PATH,
            scheduler_backend="mixed",
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=1024,
            max_num_batched_tokens=512,
            max_num_seqs=4,
            gpu_memory_utilization=0.7,
        )
        config = llm.model_runner.config
        recorder = GreedyRecorder()
        cached_at_first_step = {}
        llm.model_runner.sampler = recorder
        token = llm.tokenizer.encode("你好", add_special_tokens=False)[0]
        prompt_a = [token] * 32
        prompt_b = [token] * 768
        params_a = SamplingParams(temperature=0.6, max_tokens=6, ignore_eos=True)
        params_b = SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True)

        def attach_scheduler(scheduler):
            llm.scheduler = scheduler
            original_schedule = scheduler.schedule
            mixed_steps = []

            def schedule():
                seqs, mode = original_schedule()
                recorder.seqs = seqs
                for seq in seqs:
                    cached_at_first_step.setdefault(seq.seq_id, seq.num_cached_tokens)
                phases = [(seq.is_prefill, seq.num_scheduled_tokens) for seq in seqs]
                if any(phase for phase, _ in phases) and any(not phase for phase, _ in phases):
                    mixed_steps.append(phases)
                return seqs, mode

            scheduler.schedule = schedule
            return mixed_steps

        def add(prompt, params):
            llm.add_request(prompt, params)
            return llm.scheduler.waiting[-1]

        try:
            attach_scheduler(Scheduler(config))
            isolated_a = add(prompt_a, params_a)
            while not llm.is_finished():
                llm.step()

            attach_scheduler(Scheduler(config))
            isolated_b = add(prompt_b, params_b)
            while not llm.is_finished():
                llm.step()

            mixed_steps = attach_scheduler(UnifiedScheduler(config))
            mixed_a = add(prompt_a, params_a)
            while mixed_a.num_completion_tokens < 3:
                llm.step()
            mixed_b = add(prompt_b, params_b)
            while not llm.is_finished():
                llm.step()

            self.assertTrue(mixed_steps, "未观察到同一轮 Prefill 和 Decode")
            print("实际混合批次（是否 Prefill，Token 数）：", mixed_steps)

            # 使用真实 Token 时间反馈，验证预算收缩时仍能正确生成
            config.initial_prefill_budget = 256
            config.target_itl_ms = 1.0  # 测试中强制触发收缩，不作为性能目标
            budget_steps = attach_scheduler(BudgetedScheduler(config))
            budget_a = add(prompt_a, params_a)
            while budget_a.num_completion_tokens < 3:
                llm.step()
            budget_b = add(prompt_b, params_b)
            budget_trace = []
            while not llm.is_finished():
                llm.step()
                budget_trace.append(dict(llm.last_step_stats))
            self.assertTrue(budget_steps)
            self.assertTrue(any(
                step["next_prefill_budget"] < step["prefill_budget"]
                for step in budget_trace
            ))
            self.assertTrue(all(
                step["prefill_tokens"] <= step["prefill_budget"]
                for step in budget_trace
            ))
            print("预算反馈：", [(s["prefill_budget"], s["next_prefill_budget"])
                              for s in budget_trace if s["prefill_tokens"]])

            # 先计算 C 的独立参考，再在混合调度中使用已缓存的 256 Token 前缀
            other = llm.tokenizer.encode("世界", add_special_tokens=False)[0]
            prompt_c = [token] * 256 + [other] * 44
            params_c = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
            attach_scheduler(Scheduler(config))
            isolated_c = add(prompt_c, params_c)
            while not llm.is_finished():
                llm.step()

            prefix_steps = attach_scheduler(UnifiedScheduler(config))
            seed = add(prompt_c, params_c)
            while not llm.is_finished():
                llm.step()
            cached_a = add(prompt_a, params_a)
            while cached_a.num_completion_tokens < 3:
                llm.step()
            mixed_c = add(prompt_c, params_c)
            while not llm.is_finished():
                llm.step()
            self.assertEqual(cached_at_first_step[mixed_c.seq_id], 256)
            self.assertTrue(prefix_steps)
            print("C 命中前缀 Token 数：", cached_at_first_step[mixed_c.seq_id])

            for name, isolated, mixed in [
                ("A", isolated_a, mixed_a),
                ("B", isolated_b, mixed_b),
                ("A（动态预算）", isolated_a, budget_a),
                ("B（动态预算）", isolated_b, budget_b),
                ("C（前缀缓存）", isolated_c, mixed_c),
            ]:
                self.assertEqual(isolated.completion_token_ids, mixed.completion_token_ids)
                expected = recorder.logits[isolated.seq_id]
                actual = recorder.logits[mixed.seq_id]
                self.assertEqual(len(expected), len(actual))
                for i, (left, right) in enumerate(zip(expected, actual)):
                    mae = (left - right).abs().mean().item()
                    cosine = torch.nn.functional.cosine_similarity(left, right, dim=0).item()
                    print(f"{name} 第 {i + 1} 个 Token：logits 平均误差 {mae:.5f}，余弦相似度 {cosine:.6f}")
                    self.assertLess(mae, 0.2)
                    self.assertGreater(cosine, 0.998)
        finally:
            atexit.unregister(llm.exit)
            llm.exit()


if __name__ == "__main__":
    unittest.main(verbosity=2)
