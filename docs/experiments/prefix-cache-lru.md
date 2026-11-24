# Prefix Cache 淘汰策略实验报告（Baseline vs FIFO vs LRU）

## 1. 问题定义与动机

原始 `nanovllm/engine/block_manager.py` 使用单个 `deque` 管理所有 `ref_count == 0` 的空闲物理块，在实际 Serving 负载下存在两类系统缺陷：

1. **未哈希尾块误触发缓存淘汰**：绝大多数请求长度不是 `block_size=256` 的整数倍，请求结束时会把未填满的尾块（`hash == -1`）与已哈希前缀块一起 `append` 到 `deque` 队尾。当短请求频繁到来并释放未哈希块时，无复用价值的尾块排在已缓存前缀块后面，导致下一次分配从队头 `popleft()` 时**提前淘汰仍有价值的前缀缓存块**。
2. **复用线性扫描与策略不可插拔**：复用空闲缓存块调用 `deque.remove(block_id)`，时间复杂度为 $O(N)$；同时并发写入相同前缀 Hash 产生的陈旧（shadowed）重复块不会被降级回收。

## 2. 核心设计

在保留原始 `BlockManager` 为默认 `baseline` 的前提下，新增：

- `nanovllm/experimental/cache/eviction_policy.py`：抽象两级空闲池接口 `EvictionPolicy`，严格遵守 **“未哈希/失效陈旧块优先回收 → 仅在无优先回收块时按策略淘汰活跃缓存块”** 的顺序：
  - `LRUEvictionPolicy`：双 `OrderedDict` 实现，$O(1)$ 完成 `add / remove / pop_victim / demote_to_unhashed`；配合 `deallocate()` 反序释放，使同一序列中更深的后缀块先于浅层根前缀块被淘汰。
  - `FIFOEvictionPolicy`：基于最小堆 + 惰性删除严格按缓存首次写入顺序淘汰活跃块，用于验证“忽视访问局部性”对链式哈希前缀缓存的影响。
- `nanovllm/experimental/cache/block_manager_lru.py`：实现 `PolicyBlockManager`、`NoCacheBlockManager`（`none` 对照组）及工厂 `build_block_manager(config)`，通过 `Config.cache_eviction_policy`（`baseline` / `none` / `fifo` / `lru`）无侵入切换。

## 3. 正确性与生命周期门禁

运行全套 29 项单元测试（`tests/unit/test_block_manager.py`、`tests/unit/test_prefix_cache.py`、`tests/unit/test_cache_metrics.py` 等）：

- 校验核心不变量：`len(used) + len(free) == num_blocks`、`used ∩ free == ∅`、`used` 块均满足 `ref_count >= 1`、`free` 块均满足 `ref_count == 0`、`hash_to_block_id` 无陈旧脏映射。
- 覆盖边界场景：强制哈希碰撞时的 Token 内容校验防御、`ref_count > 0` 不可淘汰保护、未哈希尾块优先回收（直接复现 `baseline` 误淘汰 vs `lru`/`fifo` 零误淘汰）、`LRU` vs `FIFO` 热点保留分化、深层后缀先于根前缀淘汰、并发重复写入陈旧块降级、以及紧张显存下 Scheduler 抢占恢复循环。
- GPU 端到端一致性：所有基准实验均开启准贪心采样（`temperature=1e-5` + 固定随机种子），`none / baseline / fifo / lru` 四种策略在全部请求上实现 **100% 逐 Token 精确一致**。

## 4. 实验环境与结果

- **硬件与软件**：NVIDIA GeForce RTX 3060 Laptop GPU (6 GiB, Ampere `sm_86`), WSL2 Ubuntu, PyTorch `2.14.1+cu130`, Triton `3.8.0`, FlashAttention `2.8.4`
- **模型与参数**：`Qwen3-0.6B` (BF16), `enforce_eager=True`, `block_size=256`, `max_model_len=2048`

### 4.1 实验 A：共享前缀比例扫描（总长 1664 Token，每档 12 次轮换测量）

| 共享前缀长度 | 共享比例 | No-Cache TTFT P50 | LRU TTFT P50 | TTFT P50 降幅 | 平均复用 Token | 输出一致率 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0 / 1664` | 0.0% | 90.75 ms | 93.02 ms | -2.5% | 0 | 12/12 |
| `512 / 1664` | 30.8% | 91.39 ms | 72.60 ms | **-20.6%** | 512 | 12/12 |
| `1024 / 1664` | 61.5% | 92.89 ms | 46.81 ms | **-49.6%** | 1024 | 12/12 |
| `1536 / 1664` | 92.3% | 91.78 ms | **36.89 ms** | **-59.8%** | 1536 | 12/12 |

> **关键观察**：当共享比例达到 `92.3%`（仅剩 128 个新 Token 待计算）时，Warm TTFT P50 降至 `36.89 ms`（`-59.8%`），但不再继续线性下降——在 Eager 模式下，Qwen3-0.6B（28 层）单次前向存在约 **`35～38 ms` 的固定 Python Dispatch + CUDA Kernel Launch 底噪**。这一实测现象直接引出了下一阶段（M2）的 Profiling 与 CUDA Graph 优化方向。

### 4.2 实验 B：紧 KV Cache 容量混合流四策略对比（`cache_blocks=14`，50 条请求，3 轮轮换）

负载组成：2 条热点前缀冷启动 + 24 条热点系统提示词请求（1024 共享前缀 + 64 唯一尾块，理论最大可复用 `96` 块 = `24,576` Token）+ 6 条冷长请求（1088 Token）+ 18 条短对话（64 Token 未哈希块）。

| 策略 | Block 查表命中率 | 实际复用块 / 理论上限 | 实际复用 Token | 缓存淘汰次数 | 热请求 TTFT P50 | 相对 No-Cache 降幅 | 输出一致率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `none` (No-Cache) | 0.0% | 0 / 96 | 0 | 0 | 61.80 ms | — | 50/50 |
| `fifo` | 83.7% | 72 / 96 | 18,432 | 43 | 39.57 ms | -36.0% | 50/50 |
| `baseline` (`deque`) | 80.8% | 84 / 96 | 21,504 | 33 | 37.02 ms | -40.1% | 50/50 |
| **`lru`** | **92.3%** | **96 / 96 (100%)** | **24,576** | **19** | **37.53 ms** | **-39.3%** | **50/50** |

## 5. 分析、Trade-off 与局限性

1. **为什么 `FIFO` 反而比 `baseline` 复用更少？**  
   Prefix Cache 使用**链式哈希**（块 `i` 依赖块 `0..i-1`）。`FIFO` 严格按首次写入顺序淘汰，导致冷请求到来时优先淘汰最早写入的热点根块 `B0`，**单个根块被淘汰会直接引发后续 3 个热点块的级联失配**（一次失配损失 1024 Token），因此淘汰次数高达 `43` 次、仅复用 `72/96` 块。
2. **为什么 `LRU` 在缓存效率上显著领先 `baseline`？**  
   两级空闲池彻底消除了短请求未哈希尾块对热点缓存的误挤出，同时刷新热点块的 recency，使 24 条热请求实现 **`96/96`（100%）满复用**，相比 `baseline` 减少 **`42.4%` 淘汰次数（33 → 19）**、多复用 **`+14.3%` Token（21,504 → 24,576）**；在更短前缀（`576` Token、`cache_blocks=8`）的紧缓存测试中，`LRU` 同样将命中率从 `baseline` 的 `64.3%` 提升至 `85.7%`（复用 Token `+33.3%`，淘汰 `-60.9%`）。
3. **当前实验局限（诚实口径）**：
   - 在单卡移动版 RTX 3060 上连续多轮运行时存在轻度功耗/频率波动（第 2 轮 `lru` 出现约 4～5 ms 全局漂移），导致 3 轮均值中 `lru` 与 `baseline` 的 P50 相近（首轮无漂移时 `lru` 总耗时 `2037.5 ms` < `baseline` `2077.9 ms`）。
   - 实验 B 统计的 26 条热请求包含前 2 条冷启动请求（占 `7.7%`），因此热请求 P95 落在冷启动样本上（约 `61 ms`），反映冷热混合分布而非纯 Warm 尾延迟。
   - 仅当输入较长（GPU Prefill 计算时间明显超过 Eager ~35 ms Launch 底噪）时，缓存复用才会转化为显著端到端 TTFT 降低；在短输入（如 576 Token、0.6B 模型）下，收益主要体现为减少 GPU 算力占用而非墙钟延迟。

## 6. 复现命令

```bash
# 1. 单元测试
PYTHONPATH=. python -m unittest discover -s tests/unit -v

# 2. GPU 基准测试（将路径替换为本地 Qwen3-0.6B 目录）
export NANOVLLM_TEST_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
PYTHONPATH=. python benchmarks/benchmark_prefix_cache.py --model "$NANOVLLM_TEST_MODEL" --mode eval
```
