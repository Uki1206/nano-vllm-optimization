import hashlib
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import torch

# 使用本机已经下载的模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from nanovllm import LLM, SamplingParams


def run_case(llm, body_token, pair_id, inject_b):
    """
    运行一组对照实验：A 生成 24 个 Token；
    如果 inject_b 为 True，在 A 输出第 5 个 Token 后加入 B。
    """
    assert llm.is_finished()

    # 每对实验使用相同的随机种子，保证 A 的开始条件一致
    torch.manual_seed(20261110 + pair_id)

    prompt_a = [body_token] * 32
    params_a = SamplingParams(temperature=0.6, max_tokens=24, ignore_eos=True)
    params_b = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

    # 每次 B 使用不同前缀，避免同一 B 的完整前缀缓存被重复命中
    prefix_b = llm.tokenizer.encode(
        f"case-{pair_id:03d}",
        add_special_tokens=False,
    )
    prompt_b = (prefix_b + [body_token] * 768)[:768]

    start_time = perf_counter()
    llm.add_request(prompt_a, params_a)
    request_a = llm.scheduler.waiting[-1]

    token_times = []
    affected_steps = []
    injected = False

    while not llm.is_finished():
        # 对照组没有 B；干扰组在 A 的第 5 个输出 Token 后加入 B
        if inject_b and not injected and request_a.num_completion_tokens == 5:
            llm.add_request(prompt_b, params_b)
            injected = True

        previous_count = request_a.num_completion_tokens
        step_start = perf_counter()

        # 正数表示本轮 Prefill 的 Token 数；负数表示 Decode 请求数
        _, scheduled_tokens = llm.step()
        step_end = perf_counter()

        if inject_b and injected and previous_count == 5:
            affected_steps.append({
                "phase": "prefill" if scheduled_tokens > 0 else "decode",
                "scheduled_tokens": abs(scheduled_tokens),
                "step_duration_s": step_end - step_start,
            })

        if request_a.num_completion_tokens > previous_count:
            token_times.append(step_end)

    assert len(token_times) == 24
    assert not inject_b or injected

    # 比较 A 的第 5 到第 6 个输出 Token 之间的间隔
    gap_s = token_times[5] - token_times[4]
    itls_s = [
        token_times[i] - token_times[i - 1]
        for i in range(1, len(token_times))
    ]

    return {
        "pair": pair_id,
        "case": "with_b" if inject_b else "a_only",
        "output_tokens_a": len(token_times),
        "ttft_a_s": token_times[0] - start_time,
        "gap_after_token_5_s": gap_s,
        "token_offsets_a_s": [t - start_time for t in token_times],
        "itls_a_s": itls_s,
        "b_prefix_token_ids": prefix_b if inject_b else None,
        "affected_steps": affected_steps,
    }


def main():
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
        "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    )

    print("正在加载模型，请稍候...")
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
    )

    body_token = llm.tokenizer.encode("你好", add_special_tokens=False)[0]

    # 先预热单请求和加入 B 两种情况，不将预热结果计入实验
    print("正在预热...")
    run_case(llm, body_token, pair_id=0, inject_b=False)
    run_case(llm, body_token, pair_id=0, inject_b=True)

    runs = []

    # 交替调整组内运行顺序，减轻固定先后顺序的影响
    for pair_id in range(1, 4):
        order = [False, True] if pair_id % 2 else [True, False]
        for inject_b in order:
            result = run_case(llm, body_token, pair_id, inject_b)
            runs.append(result)

            label = "A + B" if inject_b else "仅 A"
            gap_ms = result["gap_after_token_5_s"] * 1000
            print(f"第 {pair_id} 组，{label}：第 5～6 个 Token 间隔 {gap_ms:.2f} 毫秒")

            if inject_b:
                steps = [
                    f'{step["phase"]}:{step["scheduled_tokens"]}'
                    for step in result["affected_steps"]
                ]
                print(f"  此间调度步骤：{steps}")

    a_only = [r["gap_after_token_5_s"] for r in runs if r["case"] == "a_only"]
    with_b = [r["gap_after_token_5_s"] for r in runs if r["case"] == "with_b"]

    print(f"仅 A，中位数：{statistics.median(a_only) * 1000:.2f} 毫秒")
    print(f"A + B，中位数：{statistics.median(with_b) * 1000:.2f} 毫秒")

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_dir, text=True
    ).strip()

    with open(__file__, "rb") as source_file:
        script_sha256 = hashlib.sha256(source_file.read()).hexdigest()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_worktree_dirty": bool(git_status),
        "benchmark_script_sha256": script_sha256,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
        },
        "workload": {
            "model_path": model_path,
            "body_token_id": body_token,
            "a_prompt_tokens": 32,
            "a_output_tokens": 24,
            "b_prompt_tokens": 768,
            "b_output_tokens": 1,
            "inject_after_a_tokens": 5,
            "max_num_batched_tokens": 512,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.7,
            "temperature": 0.6,
            "ignore_eos": True,
            "warmups_per_case": 1,
            "measured_pairs": 3,
            "random_seed_base": 20261110,
        },
        "runs": runs,
    }

    result_dir = os.path.join(repo_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    file_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    result_path = os.path.join(result_dir, f"scheduler_interference_{file_time}.json")

    with open(result_path, "x", encoding="utf-8") as result_file:
        json.dump(report, result_file, ensure_ascii=False, indent=2)

    print(f"实验结果已保存到：{result_path}")


if __name__ == "__main__":
    main()
