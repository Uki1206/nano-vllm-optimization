import argparse
import hashlib
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.experimental.schedulers.unified_scheduler import UnifiedScheduler


def run_case(llm, scheduler_class, body_token, pair, b_prompt_tokens):
    # 每次使用新的 BlockManager，避免某一组因复用另一组的前缀缓存获益
    llm.scheduler = scheduler_class(llm.model_runner.config)
    torch.manual_seed(20261110 + pair)

    prompt_a = [body_token] * 32
    prefix = llm.tokenizer.encode(f"case-{pair:03d}", add_special_tokens=False)
    prompt_b = (prefix + [body_token] * b_prompt_tokens)[:b_prompt_tokens]
    params_a = SamplingParams(temperature=0.6, max_tokens=16, ignore_eos=True)
    params_b = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

    llm.add_request(prompt_a, params_a)
    request_a = llm.scheduler.waiting[-1]
    token_times = []
    affected_steps = []
    injected = False
    b_arrival = b_finish = None

    torch.cuda.synchronize()
    start = perf_counter()
    while not llm.is_finished():
        if not injected and request_a.num_completion_tokens == 5:
            b_arrival = perf_counter()
            llm.add_request(prompt_b, params_b)
            request_b = llm.scheduler.waiting[-1]
            injected = True

        previous_count = request_a.num_completion_tokens
        step_start = perf_counter()
        llm.step()
        torch.cuda.synchronize()
        step_end = perf_counter()

        if injected and previous_count == 5:
            affected_steps.append({
                **llm.last_step_stats,
                "duration_ms": (step_end - step_start) * 1000,
            })
        if request_a.num_completion_tokens > previous_count:
            token_times.append(step_end)
        if injected and request_b.is_finished and b_finish is None:
            b_finish = step_end

    assert injected and len(token_times) == 16 and b_finish is not None
    return {
        "gap_a_5_to_6_ms": (token_times[5] - token_times[4]) * 1000,
        "ttft_b_ms": (b_finish - b_arrival) * 1000,
        "completion_a_ms": (token_times[-1] - start) * 1000,
        "affected_steps": affected_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="对比 baseline 和 mixed 调度")
    parser.add_argument("--model", required=True, help="本地模型目录")
    parser.add_argument("--pairs", type=int, default=3, help="测量组数，默认 3")
    parser.add_argument(
        "--b-prompt-tokens",
        type=int,
        default=768,
        help="请求 B 的输入 Token 数，默认 768",
    )

    args = parser.parse_args()

    if args.pairs < 1:
        parser.error("--pairs 必须大于 0")

    if not 1 <= args.b_prompt_tokens <= 1023:
        parser.error("--b-prompt-tokens 必须在 1 到 1023 之间")

    llm = LLM(
        args.model,
        scheduler_backend="mixed",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
    )
    token = llm.tokenizer.encode("你好", add_special_tokens=False)[0]
    strategies = {"baseline": Scheduler, "mixed": UnifiedScheduler}

    print("正在预热两种策略...")
    for scheduler_class in strategies.values():
        run_case(llm, scheduler_class, token, 0, args.b_prompt_tokens)

    results = {"baseline": [], "mixed": []}
    for pair in range(1, args.pairs + 1):
        order = ["baseline", "mixed"] if pair % 2 else ["mixed", "baseline"]
        for name in order:
            result = run_case(llm, strategies[name], token, pair, args.b_prompt_tokens)
            results[name].append(result)
            print(f'第 {pair} 组 {name}：A 的第 5～6 个 Token 间隔 '
                  f'{result["gap_a_5_to_6_ms"]:.2f} ms；'
                  f'B 的首 Token 耗时 {result["ttft_b_ms"]:.2f} ms')
            print("  此间的调度步骤：", result["affected_steps"])

    for name in strategies:
        gap = statistics.median(r["gap_a_5_to_6_ms"] for r in results[name])
        ttft = statistics.median(r["ttft_b_ms"] for r in results[name])
        print(f"{name} 中位数：A 间隔 {gap:.2f} ms，B 首 Token {ttft:.2f} ms")

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True).strip()
    with open(__file__, "rb") as script:
        script_hash = hashlib.sha256(script.read()).hexdigest()

    report = {
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_status),
        "benchmark_script_sha256": script_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "model": args.model,
        "workload": {
            "a_prompt_tokens": 32,
            "a_output_tokens": 16,
            "b_prompt_tokens": args.b_prompt_tokens,
            "b_output_tokens": 1,
            "inject_after_a_tokens": 5,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.7,
            "warmups_per_strategy": 1,
            "pairs": args.pairs,
        },
        "results": results,
    }
    result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(result_dir, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("mixed_batch_%Y%m%d_%H%M%S_%f.json")
    result_path = os.path.join(result_dir, name)
    with open(result_path, "x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"实验结果已保存：{result_path}")


if __name__ == "__main__":
    main()
