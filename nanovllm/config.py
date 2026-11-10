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
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.scheduler_backend not in ("baseline", "mixed"):
            raise ValueError(f"Unknown scheduler backend: {self.scheduler_backend}")
        if self.scheduler_backend == "mixed":
            if self.tensor_parallel_size != 1 or not self.enforce_eager:
                raise ValueError("mixed scheduler requires tensor_parallel_size=1 and enforce_eager=True")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)