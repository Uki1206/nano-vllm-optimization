# Mixed Batch 的 Prefill Budget 控制

## 目的

一轮的总 Token Budget 保持不变；限制其中所有 Prefill 请求合计可使用的 Token。
Decode 在调度时先选入，Prefill 使用剩余的总预算和 Prefill 上限中较小的值。
这是一种权衡：较小的 Prefill Chunk 可能减少单轮对 Decode 的干扰，但通常增加长请求
Prefill 的轮数及等待时间。是否有效必须测量。

```text
总预算 = 512，本轮已选 Decode = 3，Prefill 上限 = 256

Prefill 允许 = min(512 - 3, 256) = 256
整轮最多 = 3 + 256 = 259 Token
```

可在一个前向周期中调度多个 Prefill；256 是这些请求共享的总上限。

## 三种模式

- `scheduler_backend="mixed"`：现有 Mixed，Prefill 使用总预算的剩余部分。
- `scheduler_backend="budgeted", prefill_budget_mode="fixed"`：固定 Prefill 上限。
- `scheduler_backend="budgeted", prefill_budget_mode="adaptive"`：根据输出间隔调整上限。

`baseline` 仍为默认调度器。预算版本和原 Mixed 一样，目前只支持单卡 eager。

```python
llm = LLM(
    model_path,
    scheduler_backend="budgeted",
    prefill_budget_mode="adaptive",  # 设为 fixed 可关闭动态调整
    tensor_parallel_size=1,
    enforce_eager=True,
    max_num_batched_tokens=512,
    min_prefill_budget=64,
    max_prefill_budget=512,
    initial_prefill_budget=256,
    target_itl_ms=50,
    itl_ema_alpha=0.2,
)
```

`target_itl_ms` 是实验目标，不保证实现该值；参数需要结合机器和负载测试。

## 实际反馈链路

1. `LLMEngine.step` 调度前记录是否有 Prefill 等待。
2. 模型执行并完成采样后，`BudgetedScheduler.observe_step` 记录实际输出时间。
3. 只对同一请求有两个相邻输出时间的情况计算 ITL；第一 Token 和未完成的 Prefill
   Chunk 不作为 ITL 样本。已结束请求的时间记录会被删除。
4. 同一轮有多个有效样本时，取该轮最大 ITL 更新一次 EMA。这个样本包括请求在引擎
   调用之间等待的时间，不是纯 GPU kernel 耗时，也不是网络端到端耗时。
5. 有 Prefill 等待且 EMA 大于目标时，预算乘以 0.8；EMA 低于目标的 80% 时，
   预算增加 32；中间区域保持不变。结果被限制在最小和最大预算内。
6. 控制器输出用于下一轮；本轮的预算不会在执行过程中改变。

`llm.last_step_stats` 中记录了本轮 `prefill_budget`、调整后
`next_prefill_budget` 和 `ema_itl_ms`。固定模式的预算不随 ITL 改变。

这种规则是可解释的实验原型。它没有保证目标 ITL 或等待上界，也没有 Prefill aging、
完整生产级多请求公平性、在线到达率控制和稳定性保证。没有有效 ITL 样本时不会调整。

## 验证

```bash
PYTHONPATH=. python -m unittest discover -s tests/unit -p 'test_prefill_budget.py' -v
PYTHONPATH=. python -m unittest discover -s tests/unit -p 'test_unified_scheduler.py' -v
```

GPU 验证（需要先将环境变量设置为本地 Qwen3-0.6B 模型路径）：

```bash
NANOVLLM_TEST_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
export NANOVLLM_TEST_MODEL
PYTHONPATH=. python tests/integration/test_mixed_batch.py
```

## 三种策略对照

```bash
PYTHONPATH=. python benchmarks/benchmark_budget_controller.py \\
  --model "$NANOVLLM_TEST_MODEL" --pairs 3 \\
  --initial-budget 256 --target-itl-ms 50
```

每次运行包含一个 32 Token 输入、生成 24 Token 的 A，以及在 A 生成第 5、7、9
个 Token 后依次加入的三个 768 Token 输入、只生成 1 Token 的 B。每组使用同一
模型、相同负载，且轮换三种策略的执行顺序；每次重建 BlockManager。结果保存到
`results/prefill_budget_*.json`，包括逐步预算及样本。

这是由 A 的输出进度触发的合成到达，不是固定到达率的 Open-loop 负载；少量重复只
能用于观察机制，不应据此得出 P99、生产吞吐量或普遍性能提升结论。
