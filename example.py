import os

# 使用本地缓存的模型，避免运行时联网下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from nanovllm import LLM, SamplingParams

def main():
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
        "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    )

    print("正在加载模型，请稍候...")

    # 创建推理引擎
    llm = LLM(
        model_path,
        enforce_eager=True,         # 先使用普通执行方式，便于理解和排查问题
        tensor_parallel_size=1,     # 使用一张 GPU
        max_model_len=512,          # 本次实验允许的最大序列长度
        max_num_batched_tokens=512,  # 每轮最多处理的 Token 数
        max_num_seqs=4,             # 每轮最多调度的请求数
        gpu_memory_utilization=0.7, # 为 KV Cache 分配显存时使用的比例
    )

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=16
    )


    # 提交一个请求，并等待生成完成
    prompt = "Hello, how are you?"
    outputs = llm.generate(
        [prompt],
        sampling_params,
        use_tqdm=False,
    )

    # 一个 prompt 对应一个输出结果
    output = outputs[0]

    print(f"输入：{prompt}")
    print(f"生成文本：{output['text']}")
    print(f"生成 Token 数：{len(output['token_ids'])}")
    print(f"生成的 Token ID:{output['token_ids']}")

if __name__ == "__main__":
    main()