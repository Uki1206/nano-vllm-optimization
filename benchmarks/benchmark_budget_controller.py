import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.experimental.schedulers.budgeted_scheduler import BudgetedScheduler
from nanovllm.experimental.schedulers.unified_scheduler import UnifiedScheduler


def run_case(llm, strategy, token, pair):
    config = copy.copy(llm.model_runner.config)
    config.prefill_budget_mode = strategy
    if strategy == "mixed":
        llm.scheduler = UnifiedScheduler(config)
    else:
        llm.scheduler = BudgetedScheduler(config)
    torch.manual_seed(20261112 + pair)

    params_a = SamplingParams(temperature=0.6, max_tokens=24, ignore_eos=True)
    params_b = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)
    llm.add_request([token] * 32, params_a)
    request_a = llm.scheduler.waiting[-1]
    token_times = []
    pending_arrivals = [5, 7, 9]
    requests_b = []
    timeline = []

    torch.cuda.synchronize()
    start = perf_counter()
    while not llm.is_finished():
        if pending_arrivals and request_a.num_completion_tokens >= pending_arrivals[0]:
            arrival_at = pending_arrivals.pop(0)
            prefix = llm.tokenizer.encode(f"pair-{pair}-arrival-{arrival_at}", add_special_tokens=False)
            prompt_b = (prefix + [token] * 768)[:768]
            arrival = perf_counter()
            llm.add_request(prompt_b, params_b)
            requests_b.append({"seq": llm.scheduler.waiting[-1], "arrival": arrival, "finish": None})

        previous_count = request_a.num_completion_tokens
        had_pending_b = any(request["finish"] is None for request in requests_b)
        llm.step()
        torch.cuda.synchronize()
        now = perf_counter()
        if request_a.num_completion_tokens > previous_count:
            token_times.append(now)
        for request in requests_b:
            if request["seq"].is_finished and request["finish"] is None:
                request["finish"] = now

        if had_pending_b:
            timeline.append({
                "a_output_tokens": request_a.num_completion_tokens,
                **llm.last_step_stats,
            })

    assert not pending_arrivals and len(token_times) == 24
    assert len(requests_b) == 3 and all(request["finish"] is not None for request in requests_b)
    return {
        "a_itl_ms": [(token_times[i] - token_times[i - 1]) * 1000 for i in range(1, len(token_times))],
        "b_ttft_ms": [(request["finish"] - request["arrival"]) * 1000 for request in requests_b],
        "a_completion_ms": (token_times[-1] - start) * 1000,
        "timeline": timeline,
    }


def main():
    parser = argparse.ArgumentParser(description="比较 Mixed 的无上限、固定上限和动态 Prefill Budget")
    parser.add_argument("--model", required=True, help="本地模型目录")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--initial-budget", type=int, default=256)
    parser.add_argument("--target-itl-ms", type=float, default=50)
    args = parser.parse_args()
    if args.pairs < 1 or not 64 <= args.initial_budget <= 512 or args.target_itl_ms <= 0:
        parser.error("要求 pairs >= 1，initial-budget 在 64 到 512 之间，target-itl-ms 为正数")

    llm = LLM(
        args.model,
        scheduler_backend="budgeted",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
        min_prefill_budget=64,
        max_prefill_budget=512,
        initial_prefill_budget=args.initial_budget,
        target_itl_ms=args.target_itl_ms,
    )
    token = llm.tokenizer.encode("你好", add_special_tokens=False)[0]
    strategies = ["mixed", "fixed", "adaptive"]
    print("正在预热...")
    for name in strategies:
        run_case(llm, name, token, 0)

    results = {name: [] for name in strategies}
    for pair in range(1, args.pairs + 1):
        shift = (pair - 1) % len(strategies)
        order = strategies[shift:] + strategies[:shift]
        for name in order:
            run = run_case(llm, name, token, pair)
            results[name].append(run)
            print(f'第 {pair} 组 {name}：A 最大 ITL {max(run["a_itl_ms"]):.2f} ms，'
                  f'B 首 Token {", ".join(f"{v:.2f}" for v in run["b_ttft_ms"])} ms')

    for name, runs in results.items():
        a_itls = [v for run in runs for v in run["a_itl_ms"]]
        b_ttfts = [v for run in runs for v in run["b_ttft_ms"]]
        print(f"{name}：A ITL 中位数 {statistics.median(a_itls):.2f} ms，"
              f"每次运行的 A 最大 ITL 中位数 {statistics.median(max(run['a_itl_ms']) for run in runs):.2f} ms，"
              f"B TTFT 中位数 {statistics.median(b_ttfts):.2f} ms")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    with open(__file__, "rb") as file:
        script_hash = hashlib.sha256(file.read()).hexdigest()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "benchmark_script_sha256": script_hash,
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "model": args.model,
        "workload": {
            "a_prompt_tokens": 32, "a_output_tokens": 24,
            "b_prompt_tokens": 768, "b_output_tokens": 1,
            "b_arrivals_after_a_tokens": [5, 7, 9],
            "max_num_batched_tokens": 512, "initial_prefill_budget": args.initial_budget,
            "min_prefill_budget": 64, "max_prefill_budget": 512,
            "target_itl_ms": args.target_itl_ms, "ema_alpha": 0.2,
            "pairs": args.pairs, "warmups_per_strategy": 1,
        },
        "results": results,
    }
    result_dir = os.path.join(repo, "results")
    os.makedirs(result_dir, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("prefill_budget_%Y%m%d_%H%M%S_%f.json")
    path = os.path.join(result_dir, filename)
    with open(path, "x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"实验结果已保存：{path}")


if __name__ == "__main__":
    main()
