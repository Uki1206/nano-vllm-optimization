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
        start_time = perf_counter()

        outputs = llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
        )

        elapsed = perf_counter() - start_time
        latencies.append(elapsed)

        output = outputs[0]
        token_count = len(output["token_ids"])

        print(
            f"第 {i + 1} 次："
            f"总耗时 {elapsed:.3f} 秒，"
            f"生成 {token_count} 个 Token"
        )
        print(f"生成文本：{output['text']!r}")

    average_latency = sum(latencies) / len(latencies)
    print(f"\n三次平均总耗时：{average_latency:.3f} 秒")


if __name__ == "__main__":
    main()