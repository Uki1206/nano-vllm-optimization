from nanovllm.engine.scheduler import Scheduler
from nanovllm.experimental.schedulers.unified_scheduler import UnifiedScheduler


SCHEDULER_REGISTRY = {
    "baseline": Scheduler,
    "mixed": UnifiedScheduler,
}


def build_scheduler(config):
    scheduler_cls = SCHEDULER_REGISTRY.get(config.scheduler_backend)
    if scheduler_cls is None:
        raise ValueError(f"Unknown scheduler backend: {config.scheduler_backend}")
    return scheduler_cls(config)
