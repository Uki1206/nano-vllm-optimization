from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import SequenceStatus


class UnifiedScheduler(Scheduler):
    """
    在一个 Token Budget 内调度 Decode 和 Prefill。

    所有请求交给变长 Attention 路径执行：Decode 的 Query 长度为 1，
    Prefill 的 Query 长度为本轮调度的 Chunk 长度。
    """

    def __init__(self, config: Config):
        super().__init__(config)
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_num_batched_tokens must be positive")
        self.prefer_solo_prefill = False

    def schedule_decode(self, scheduled_seqs, remaining, max_decode_seqs):
        decode_seqs = []

        while self.running and len(decode_seqs) < max_decode_seqs and remaining > 0:
            seq = self.running.popleft()

            # 只抢占尚未选入本轮的请求，避免释放已调度请求的 Block
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                remaining -= 1

        # 放回队尾，使未选中的 Decode 请求下轮有机会先执行
        self.running.extend(decode_seqs)
        scheduled_seqs.extend(decode_seqs)
        return remaining

    def schedule_prefill(self, scheduled_seqs, remaining):
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs and remaining > 0:
            seq = self.waiting[0]

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)

            num_tokens = seq.num_tokens - seq.num_cached_tokens
            assert num_tokens > 0
            seq.is_prefill = True
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            remaining -= seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        return remaining

    def schedule(self):
        scheduled_seqs = []
        remaining = self.max_num_batched_tokens
        can_mix = self.max_num_seqs > 1 and remaining > 1
        solo_prefill = (
            not can_mix and self.waiting and self.running and self.prefer_solo_prefill
        )

        if solo_prefill:
            # 预算不足以混合时，在 Prefill 和 Decode 之间交替
            remaining = self.schedule_prefill(scheduled_seqs, remaining)
            if not scheduled_seqs:
                remaining = self.schedule_decode(
                    scheduled_seqs, remaining, self.max_num_seqs
                )
        else:
            decode_limit = self.max_num_seqs
            if can_mix and self.waiting:
                # 至少给等待中的 Prefill 留一个请求位置和一个 Token
                decode_limit = min(decode_limit - 1, remaining - 1)

            remaining = self.schedule_decode(scheduled_seqs, remaining, decode_limit)
            remaining = self.schedule_prefill(scheduled_seqs, remaining)

        if not scheduled_seqs:
            raise RuntimeError("No request can be scheduled; check KV cache capacity")

        if not can_mix:
            self.prefer_solo_prefill = not any(seq.is_prefill for seq in scheduled_seqs)

        # True 表示选择变长 Attention 执行路径；每个请求的阶段仍由自身状态表示
        return scheduled_seqs, True
