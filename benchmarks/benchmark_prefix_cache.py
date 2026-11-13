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


def run_case(llm, shared, pair):
    # 每个实验独立 BlockManager，避免其他实验的缓存干扰
    llm.scheduler = Scheduler(llm.model_runner.config)
    tokenizer = llm.tokenizer
    a = tokenizer.encode("你好", add_special_tokens=False)[0]
    b = tokenizer.encode("世界", add_special_tokens=False)[0]
    assert a != b
    prefix = tokenizer.encode(f"prefix-{pair:03d}", add_special_tokens=False)
    shared_tokens = (prefix + [a] * 512)[:512]
    # 同长度、相同后缀，首 Token 不同使第一块无法复用
    different_tokens = shared_tokens.copy()
    different_tokens[0] = b if shared_tokens[0] != b else a
    seed_prompt = shared_tokens + [b] * 128
    target_prompt = (shared_tokens if shared else different_tokens) + [a] * 128
    params = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

    llm.add_request(seed_prompt, params)
    while not llm.is_finished():
        llm.step()

    before = llm.scheduler.block_manager.metrics.snapshot()
    torch.cuda.synchronize()
    start = perf_counter()
    llm.add_request(target_prompt, params)
    request = llm.scheduler.waiting[-1]
    while not request.is_finished:
        llm.step()
    torch.cuda.synchronize()
    end = perf_counter()
    after = llm.scheduler.block_manager.metrics.snapshot()
    delta = {key: after[key] - before[key] for key in before}

    # 当前 Block 大小为 256，512 Token 共享前缀应复用两个完整 Block
    expected = 512 if shared else 0
    assert delta["reused_tokens"] == expected, (shared, delta)
    return {"case": "shared" if shared else "different",
            "ttft_ms": (end - start) * 1000,
            "metrics": delta}


def main():
    parser = argparse.ArgumentParser(description="验证 Prefix Cache 的真实复用 Token")
    parser.add_argument("--model", required=True, help="本地模型目录")
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs 必须大于 0")

    llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1,
              max_model_len=1024, max_num_batched_tokens=512, max_num_seqs=4,
              gpu_memory_utilization=0.7)
    print("正在预热...")
    for shared in (False, True):
        run_case(llm, shared, 0)

    runs = []
    for pair in range(1, args.pairs + 1):
        order = (False, True) if pair % 2 else (True, False)
        for shared in order:
            result = run_case(llm, shared, pair)
            result["pair"] = pair
            runs.append(result)
            print(f'第 {pair} 组 {result["case"]}：TTFT {result["ttft_ms"]:.2f} ms，'
                  f'实际复用 {result["metrics"]["reused_tokens"]} Token，'
                  f'Hash 查找/命中 {result["metrics"]["lookup_count"]}/'
                  f'{result["metrics"]["hit_count"]}')

    for name in ("different", "shared"):
        values = [run["ttft_ms"] for run in runs if run["case"] == name]
        print(f"{name} TTFT 中位数：{statistics.median(values):.2f} ms")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    with open(__file__, "rb") as file:
        script_hash = hashlib.sha256(file.read()).hexdigest()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "benchmark_script_sha256": script_hash,
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "model": args.model,
        "workload": {"seed_tokens": 640, "target_tokens": 640,
                     "shared_prefix_tokens": 512, "output_tokens": 1,
                     "block_size": 256, "max_num_batched_tokens": 512,
                     "warmups_per_case": 1, "pairs": args.pairs},
        "runs": runs,
    }
    directory = os.path.join(repo, "results")
    os.makedirs(directory, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("prefix_cache_%Y%m%d_%H%M%S_%f.json")
    path = os.path.join(directory, filename)
    with open(path, "x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"结果已保存：{path}")


if __name__ == "__main__":
    main()
