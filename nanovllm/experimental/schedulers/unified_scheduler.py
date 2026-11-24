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
        can_mix = self.max_num_seqs > 1 and remaining > 1       # 判断当前配置是否允许混合调度
        solo_prefill = (
            not can_mix and self.waiting and self.running and self.prefer_solo_prefill      # 判断是否需要单独调度 Prefill，4个条件：1.当前配置不允许混合 2.有等待中的 Prefill 3.有正在执行的 Decode 4.当前轮优先 Prefill
        )# 注：solo_prefill主要用于 当不具备混合调度能力时，决定这一轮是否应该优先单独调度 Prefill，避免 Prefill 一直得不到执行机会。

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

    """
    else部分里的第一个if不生效时(也就是不能混合调度时)
    下方的两句实际上只有第一句时有效，也就是只执行decode，它就会把max_num_seqs或者remaining消耗完，
    以至于下一句prefill实际上并不会真正执行。
    这样一来也就是：当不能混合时(can_mix=False)，
    优先只会执行decode，这样会导致prefill饥饿，所以才引入了prefer_solo_prefill和solo_prefill两个变量。
    当这一轮只执行了decode时(不能混合是大前提)，prefer_solo_prefill变为True,
    下一轮solo_prefill也变为True，改为走一条单独的prefill优先的分支。
    同样，当这一轮只执行Prefill时，prefer_solo_prefill变为False,下一轮solo_prefill也变为False,
    又走原来的分支
    """