# Mixed Batch：同一轮执行 Prefill 和 Decode

## 先理解要解决的问题

原调度器优先处理等待队列中的 Prefill。假设 A 正在逐个生成 Token，
这时长请求 B 到来，B 可能占用好几轮 Prefill，A 只能等到这些轮次结束
后再生成下一个 Token。A 两个 Token 之间的间隔可能因此变长。

Mixed Batch 在同一个调度周期内选入不同阶段的请求，并把它们送入同一次
模型前向计算。例如：

```text
Token Budget = 512

原调度器：
第 1 轮：B Prefill 512 Token
第 2 轮：B Prefill 256 Token
第 3 轮：A Decode 1 Token

Mixed：
第 1 轮：A Decode 1 Token + B Prefill 511 Token
第 2 轮：A Decode 1 Token + B Prefill 257 Token
```

A 的 Decode 仍要等待同一轮模型计算完成，但不必额外等待 B 的所有
Prefill 分块执行完。实际延迟变化需要测量，不同负载下也可能变慢。

## 代码从哪里看起

1. `nanovllm/config.py`：`scheduler_backend` 默认是 `baseline`，
   设为 `mixed` 时启用实验策略。
2. `nanovllm/engine/scheduler_factory.py`：注册表把名字映射到调度器类。
3. `nanovllm/experimental/schedulers/unified_scheduler.py`：先调度
   Decode，并给等待中的 Prefill 保留位置和 Token 预算；若不能混合，
   则在两个阶段之间交替。多个 Decode 请求轮转，避免始终只选中队首。
4. `nanovllm/engine/model_runner.py`：现有的 `prepare_prefill` 可以
   接收不同 Query 长度，将 Decode 表示为长度 1，Prefill 表示为本轮
   Chunk 长度。各请求的 Block Table 和累计长度用来区分 KV Cache。
5. `nanovllm/engine/scheduler.py`：沿用原有 Block 分配、抢占和
   后处理；未完成的 Prefill Chunk 不会产生可见输出。

这个版本复用现有变长 Attention 路径。内部返回的 `is_prefill=True`
表示**使用这条执行路径**，不表示批次中每个请求都处于 Prefill 阶段。
请求的阶段由 `seq.is_prefill` 表示。

## 如何启用

```python
from nanovllm import LLM

llm = LLM(
    model_path,
    scheduler_backend="mixed",
    tensor_parallel_size=1,
    enforce_eager=True,
)
```

当前实现限定单 GPU、eager 模式；这是实验版本。它不支持混合批次的
多卡序列化和 CUDA Graph，纯 Decode 也会走变长路径，因此性能可能与
原 Decode 路径不同。移除 `scheduler_backend` 参数即可选择基线。

`LLMEngine.step()` 仍返回 `(outputs, num_tokens)`。基线沿用原来的
正负数约定；mixed 的 `num_tokens` 是本轮总 Query Token 数。请读取
`llm.last_step_stats` 中的 `prefill_tokens` 和 `decode_tokens`
区分阶段。旧的 `benchmark_scheduler.py` 假设正数就代表纯 Prefill，
不能直接用来分析 mixed。

## 测试和比较

在仓库根目录、nano-vllm 环境中运行：

```bash
PYTHONPATH=. python -m unittest discover -s tests/unit -p 'test_unified_scheduler.py' -v
```

将下面的路径替换成自己的 Qwen3-0.6B 本地模型路径。集成测试会比较
独立执行和混合执行的贪心 Token 及 logits，并覆盖一个前缀缓存命中。

```bash
export NANOVLLM_TEST_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
PYTHONPATH=. python tests/integration/test_mixed_batch.py
```

测量使用同一个加载好的模型、相同 Token Budget 和配对负载，每次重建
BlockManager，并交替调整两种策略的执行顺序：

```bash
PYTHONPATH=. python benchmarks/benchmark_mixed_batch.py --model "$NANOVLLM_TEST_MODEL"
```

输出包含 A 第 5～6 个 Token 的间隔、B 的首 Token 耗时，以及这段时间
每一轮的 Prefill/Decode Token 数。JSON 保存到本地 `results/`。默认
只有 3 组，适合观察机制；评估性能需要增加组数、不同输入长度和并发。

集成测试只是有限用例：不同批次形状会改变数值计算结果，贪心 Token
相同不意味着所有输入或随机采样输出都完全相同。
