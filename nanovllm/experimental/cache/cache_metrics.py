from dataclasses import asdict, dataclass


@dataclass
class CacheMetrics:
    # lookup / hit 按 can_allocate 的 Block 探测次数统计，重试也会累计
    lookup_count: int = 0
    hit_count: int = 0
    reused_blocks: int = 0
    reused_tokens: int = 0
    eviction_count: int = 0
    allocation_failure_count: int = 0
    preemption_count: int = 0

    def snapshot(self):
        # 返回副本，方便计算两次快照之间的指标变化
        return asdict(self)
