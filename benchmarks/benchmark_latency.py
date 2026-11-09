import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import torch

# 使用本地模型
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from nanovllm import LLM, SamplingParams


def main():
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
        "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    )

    print("正在加载模型，请稍候...")

    # 模型只加载一次，后续重复使用同一个引擎
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=16,
        ignore_eos=True,
    )

    prompt = "hello,how are you?"

    # 先运行一次预热；预热结果不计入正式测量
    print("正在预热...")
    llm.generate([prompt], sampling_params, use_tqdm=False)

    latencies = []
    run_results = []

    # 正式测量三次
    for i in range(3):
        # 本实验每次只运行一个请求
        assert llm.is_finished()

        # 记录开始提交请求的时间，包含输入文字的分词时间
        start_time = perf_counter()
        llm.add_request(prompt, sampling_params)

        # 请求刚进入 waiting 队列，取出对应的 Sequence
        seq = llm.scheduler.waiting[-1]
        token_times = []

        # 每调用一次 step，就执行一轮调度和模型推理
        while not llm.is_finished():
            previous_count = seq.num_completion_tokens

            llm.step()

            # 输出 Token 数增加，说明本轮产生了一个新 Token
            if seq.num_completion_tokens > previous_count:
                token_times.append(perf_counter())

        elapsed = perf_counter() - start_time
        latencies.append(elapsed)

        # 当前实验固定生成 16 个 Token，可以计算首 Token 和平均间隔
        assert len(token_times) >= 2

        # 计算TTFT和TPOT
        ttft = token_times[0] - start_time
        tpot = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
        # 计算每个ITL（相邻两个输出token的时间间隔,单位为秒）
        itls = [
            token_times[j] - token_times[j-1]
            for j in range(1, len(token_times))
        ]

        print(f"\n第 {i + 1} 次：")
        print(f"请求处理耗时：{elapsed:.3f} 秒")
        print(f"生成 Token 数：{len(token_times)}")
        print(f"TTFT：{ttft * 1000:.2f} 毫秒")
        print(f"TPOT：{tpot * 1000:.2f} 毫秒")

        # 将每个间隔转换成毫秒，方便观察
        itls_ms = [round(itl * 1000, 2) for itl in itls]
        print(f"每个 ITL：{itls_ms} 毫秒")
        print(f"最大 ITL：{max(itls_ms):.2f} 毫秒")

        # 保存原始数值；时间统一使用秒，暂时不做四舍五入
        run_results.append({
            "run": i + 1,
            "output_tokens": len(token_times),
            "request_latency_s": elapsed,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "itl_s": itls,
            # 每个 Token 的时间相对于本次请求开始时间
            "token_offsets_s": [
                token_time - start_time
                for token_time in token_times
            ],
        })

    average_latency = sum(latencies) / len(latencies)
    print(f"\n三次平均总耗时：{average_latency:.3f} 秒")

    # 获取项目目录和当前 Git 状态
    repo_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        text=True,
    ).strip()

    # 记录本次运行的脚本内容摘要
    with open(__file__, "rb") as source_file:
        script_sha256 = hashlib.sha256(source_file.read()).hexdigest()

    # 组织实验条件和每轮原始结果
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
            "prompt": prompt,
            "prompt_tokens": len(llm.tokenizer.encode(prompt)),
            "max_output_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "ignore_eos": sampling_params.ignore_eos,
            "warmup_runs": 1,
            "measured_runs": len(run_results),
        },
        "average_request_latency_s": average_latency,
        "runs": run_results,
    }

    # 每次运行生成一个新文件
    result_dir = os.path.join(repo_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    file_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    result_path = os.path.join(result_dir, f"latency_{file_time}.json")

    with open(result_path, "x", encoding="utf-8") as result_file:
        json.dump(report, result_file, ensure_ascii=False, indent=2)

    print(f"实验结果已保存到：{result_path}")

if __name__ == "__main__":
    main()