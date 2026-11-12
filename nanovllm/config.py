import math
import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096    # 单个 Sequence 允许的最大长度
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    scheduler_backend: str = "baseline"
    prefill_budget_mode: str = "adaptive"
    min_prefill_budget: int = 64
    max_prefill_budget: int | None = None
    initial_prefill_budget: int | None = None
    target_itl_ms: float = 50.0
    itl_ema_alpha: float = 0.2
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.scheduler_backend not in ("baseline", "mixed", "budgeted"):
            raise ValueError(f"Unknown scheduler backend: {self.scheduler_backend}")
        if self.scheduler_backend in ("mixed", "budgeted"):
            if self.tensor_parallel_size != 1 or not self.enforce_eager:
                raise ValueError("mixed and budgeted schedulers require tensor_parallel_size=1 and enforce_eager=True")
        if self.scheduler_backend == "budgeted":
            maximum = self.max_prefill_budget if self.max_prefill_budget is not None else self.max_num_batched_tokens
            initial = self.initial_prefill_budget if self.initial_prefill_budget is not None else maximum
            if self.prefill_budget_mode not in ("fixed", "adaptive"):
                raise ValueError("prefill_budget_mode must be fixed or adaptive")
            if not 1 <= self.min_prefill_budget <= initial <= maximum <= self.max_num_batched_tokens:
                raise ValueError("prefill budgets must satisfy 1 <= min <= initial <= max <= token budget")
            if not math.isfinite(self.target_itl_ms) or self.target_itl_ms <= 0:
                raise ValueError("target_itl_ms must be finite and positive")
            if not math.isfinite(self.itl_ema_alpha) or not 0 < self.itl_ema_alpha <= 1:
                raise ValueError("itl_ema_alpha must be in (0, 1]")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)