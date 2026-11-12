from nanovllm.experimental.schedulers.budget_controller import PrefillBudgetController
from nanovllm.experimental.schedulers.unified_scheduler import UnifiedScheduler


class BudgetedScheduler(UnifiedScheduler):
    """
    在 Mixed Batch 上限制每轮 Prefill 的 Token 数。

    fixed 保持固定上限；adaptive 使用实际输出间隔更新下一轮上限。
    """

    def __init__(self, config):
        super().__init__(config)
        maximum = config.max_prefill_budget
        if maximum is None:
            maximum = self.max_num_batched_tokens
        initial = config.initial_prefill_budget
        if initial is None:
            initial = maximum
        if config.prefill_budget_mode not in ("fixed", "adaptive"):
            raise ValueError("prefill_budget_mode must be fixed or adaptive")
        if not 1 <= config.min_prefill_budget <= initial <= maximum <= self.max_num_batched_tokens:
            raise ValueError("invalid prefill budget bounds")

        self.mode = config.prefill_budget_mode
        self.controller = PrefillBudgetController(
            config.min_prefill_budget,
            maximum,
            initial,
            config.target_itl_ms,
            config.itl_ema_alpha,
        )
        self.last_prefill_budget = initial
        self.last_token_times = {}

    def schedule(self):
        self.last_prefill_budget = self.controller.current_budget
        return super().schedule()

    def schedule_prefill(self, scheduled_seqs, remaining):
        # 上限作用于本轮所有 Prefill，不是每个请求各自都有一个上限
        prefill_budget = min(remaining, self.last_prefill_budget)
        unused_prefill = super().schedule_prefill(scheduled_seqs, prefill_budget)
        used_prefill = prefill_budget - unused_prefill
        return remaining - used_prefill

    def observe_step(self, seqs, previous_counts, now, waiting_prefill):
        itls_ms = []
        for seq, previous_count in zip(seqs, previous_counts):
            if seq.num_completion_tokens > previous_count:
                last_time = self.last_token_times.get(seq.seq_id)
                if last_time is not None:
                    itls_ms.append((now - last_time) * 1000)
                self.last_token_times[seq.seq_id] = now
            if seq.is_finished:
                self.last_token_times.pop(seq.seq_id, None)

        # 同一轮最多更新一次，使用该轮各请求的最大实际间隔
        if self.mode == "adaptive" and itls_ms:
            self.controller.observe_itl(max(itls_ms), waiting_prefill)
