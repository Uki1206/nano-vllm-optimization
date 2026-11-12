from nanovllm.engine.scheduler import Scheduler
from nanovllm.experimental.schedulers.unified_scheduler import UnifiedScheduler
from nanovllm.experimental.schedulers.budgeted_scheduler import BudgetedScheduler


SCHEDULER_REGISTRY = {
    "baseline": Scheduler,
    "mixed": UnifiedScheduler,
    "budgeted": BudgetedScheduler,
}


def build_scheduler(config):
    scheduler_cls = SCHEDULER_REGISTRY.get(config.scheduler_backend)
    if scheduler_cls is None:
        raise ValueError(f"Unknown scheduler backend: {config.scheduler_backend}")
    return scheduler_cls(config)
