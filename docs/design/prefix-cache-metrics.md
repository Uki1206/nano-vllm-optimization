# Prefix Cache 指标

## 为什么要统计实际复用

Prefix Cache 复用的是已经写入 KV Cache 的完整 Block。当前 BlockManager 用链式
Hash 查找候选 Block，再校验其中的 Token 内容；最后一个 Block 不参与前缀复用，
以保证请求仍有新 Token 需要计算。

查找命中不等于实际复用：请求可能因 Block 不足而无法分配；等待队列也可能多次
重试同一个请求。因此我们分别在查找、分配、淘汰和抢占的位置记录指标。

## 统计口径

指标存在于 `llm.scheduler.block_manager.metrics`，读取快照：

```python
metrics = llm.scheduler.block_manager.metrics.snapshot()
print(metrics)
```

- `lookup_count`：尝试查找完整前缀 Block 的次数。重复检查会重复计数。
- `hit_count`：Hash 映射存在且 Token 内容校验通过的次数，包含之后分配失败的检查。
- `reused_blocks`：实际挂接到 Sequence 的缓存 Block 数，包括仍被其他请求引用和已释放的缓存块。
- `reused_tokens`：实际挂接的 Block 数乘 Block 大小；重算恢复时再次复用也会累计。
- `eviction_count`：分配空闲 Block 时，移除了仍指向该 Block 的有效 Hash 映射的次数。
  普通的释放引用不会计入；覆盖同一 Hash 映射的其他路径不计入此项。
- `allocation_failure_count`：`can_allocate` 因空闲 Block 不足返回 -1 的次数；
  Decode 追加检查失败不在此计数中，同一请求重试会重复计数。
- `preemption_count`：Scheduler 实际调用抢占的次数。

命中率若用 `hit_count / lookup_count`，表示 Block 查找命中比例，而非请求命中率；
若没有查找，分母为 0，应显示为无数据。指标按 BlockManager 实例累计；新建
Scheduler 会重新开始计数。重复的前缀重算可能让累计复用 Token 数大于用户原始
Prompt Token 数，不能直接当作请求级节省比例。

## 验证

```bash
PYTHONPATH=. python -m unittest discover -s tests/unit -p 'test_cache_metrics.py' -v
```

测试覆盖引用中的块、已释放的块、未命中、命中后分配失败和重试、缓存映射淘汰、
调度抢占。指标只增加计数，不改变原有 Cache 分配和淘汰策略。

## GPU 对照实验

先运行种子请求，然后测量目标请求。两种情况目标长度都为 640 Token：

- `shared`：和种子共享 512 Token 前缀，应实际复用 2 个 Block。
- `different`：第一个 Token 不同，使首个 Block 不命中，应实际复用 0 Token。

每组重新创建 BlockManager，轮流改变两种情况的顺序；测量目标请求加入引擎到第一个
Token 产生的本地耗时。种子请求及模型加载时间不计入目标请求 TTFT。

```bash
export NANOVLLM_TEST_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
PYTHONPATH=. python benchmarks/benchmark_prefix_cache.py --model "$NANOVLLM_TEST_MODEL" --pairs 3
```

结果写入本地 `results/prefix_cache_*.json`。此实验中两种情况的输入内容存在差异，
重复次数较少，只展示该负载下的复用机制与耗时，不能代表所有负载收益。LRU 尚未实现，
需在后续独立修改并验证淘汰生命周期。
