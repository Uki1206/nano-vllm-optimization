import os
from time import perf_counter

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

        ttft = token_times[0] - start_time
        tpot = (token_times[-1] - token_times[0]) / (len(token_times) - 1)

        print(f"\n第 {i + 1} 次：")
        print(f"请求处理耗时：{elapsed:.3f} 秒")
        print(f"生成 Token 数：{len(token_times)}")
        print(f"TTFT：{ttft * 1000:.2f} 毫秒")
        print(f"TPOT：{tpot * 1000:.2f} 毫秒")

    average_latency = sum(latencies) / len(latencies)
    print(f"\n三次平均总耗时：{average_latency:.3f} 秒")


if __name__ == "__main__":
    main()