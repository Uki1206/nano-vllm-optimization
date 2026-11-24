import argparse
from copy import copy
import hashlib
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import Scheduler


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数（p 取 0~100）。"""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def reset_scheduler(llm: LLM, policy: str, num_kvcache_blocks: int | None = None):
    cfg = copy(llm.model_runner.config)
    cfg.cache_eviction_policy = policy
    if num_kvcache_blocks is not None:
        assert num_kvcache_blocks <= llm.model_runner.config.num_kvcache_blocks, (
            f"Requested {num_kvcache_blocks} blocks exceeds allocated "
            f"{llm.model_runner.config.num_kvcache_blocks}"
        )
        cfg.num_kvcache_blocks = num_kvcache_blocks
    llm.scheduler = Scheduler(cfg)


def run_single_ttft(llm: LLM, prompt_tokens: list[int], params: SamplingParams, seed: int) -> tuple[float, int]:
    """执行单条请求并测量 TTFT（ms）及首输出 Token ID。"""
    assert llm.is_finished()
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    start = perf_counter()
    llm.add_request(prompt_tokens, params)
    request = llm.scheduler.waiting[-1]
    while not request.is_finished:
        llm.step()
    torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - start) * 1000.0
    return elapsed_ms, request.completion_token_ids[0]


def make_distinct_block(seed_token: int, filler_token: int, length: int) -> list[int]:
    return [seed_token] + [filler_token] * (length - 1)


def run_pair_case(llm: LLM, shared: bool, pair: int):
    reset_scheduler(llm, "baseline")
    tokenizer = llm.tokenizer
    a = tokenizer.encode("你好", add_special_tokens=False)[0]
    b = tokenizer.encode("世界", add_special_tokens=False)[0]
    assert a != b
    prefix = tokenizer.encode(f"prefix-{pair:03d}", add_special_tokens=False)
    shared_tokens = (prefix + [a] * 512)[:512]
    different_tokens = shared_tokens.copy()
    different_tokens[0] = b if shared_tokens[0] != b else a
    seed_prompt = shared_tokens + [b] * 128
    target_prompt = (shared_tokens if shared else different_tokens) + [a] * 128
    params = SamplingParams(temperature=0.6, max_tokens=1, ignore_eos=True)

    run_single_ttft(llm, seed_prompt, params, seed=100 + pair)
    before = llm.scheduler.block_manager.metrics.snapshot()
    ttft_ms, _ = run_single_ttft(llm, target_prompt, params, seed=200 + pair)
    after = llm.scheduler.block_manager.metrics.snapshot()
    delta = {key: after[key] - before[key] for key in before}

    expected = 512 if shared else 0
    assert delta["reused_tokens"] == expected, (shared, delta)
    return {"case": "shared" if shared else "different", "ttft_ms": ttft_ms, "metrics": delta}


def run_ratio_sweep(llm: LLM, repeats: int = 12):
    """
    实验 A：固定总 Prompt 长度 1664 Token（6 个完整 256-token 块 + 128-token 尾块），
    扫描共享前缀长度 0 / 512 / 1024 / 1536 Token，对比 No-Cache 与 LRU Warm TTFT。
    """
    block_size = llm.model_runner.config.kvcache_block_size
    assert block_size == 256
    prompt_len = 1664
    tail_len = 128
    shared_lengths = [0, 512, 1024, 1536]
    params = SamplingParams(temperature=1e-5, max_tokens=1, ignore_eos=True)

    # 使用词表中真实存在的不同 Token 构造确定性请求
    vocab_tokens = [
        llm.tokenizer.encode(word, add_special_tokens=False)[0]
        for word in ("系统", "推理", "缓存", "模型", "延迟", "吞吐", "调度", "显存")
    ]
    base_prefix = [vocab_tokens[i % 4] for i in range(1536)]
    alt_prefix = [vocab_tokens[4 + (i % 4)] for i in range(1536)]

    # 预热一次完整 Prefill
    reset_scheduler(llm, "none")
    run_single_ttft(llm, base_prefix + [vocab_tokens[0]] * tail_len, params, seed=1)

    results = []
    for shared_len in shared_lengths:
        none_ttfts = []
        lru_ttfts = []
        reused_tokens_list = []
        exact_matches = 0

        for idx in range(repeats):
            query_suffix = [vocab_tokens[(idx + j) % len(vocab_tokens)] for j in range(tail_len)]
            seed_prompt = base_prefix + [vocab_tokens[7]] * tail_len
            if shared_len == 0:
                target_prompt = alt_prefix + query_suffix
            else:
                target_prompt = base_prefix[:shared_len] + alt_prefix[shared_len:] + query_suffix
            assert len(target_prompt) == prompt_len

            # 轮换执行顺序减轻热效应漂移
            order = ("none", "lru") if idx % 2 == 0 else ("lru", "none")
            out_tokens = {}
            for policy in order:
                reset_scheduler(llm, policy)
                # 先写入 seed_prompt（不计入目标 TTFT）
                run_single_ttft(llm, seed_prompt, params, seed=1000 + idx)
                before = llm.scheduler.block_manager.metrics.snapshot()
                ttft_ms, tok = run_single_ttft(llm, target_prompt, params, seed=2000 + idx)
                after = llm.scheduler.block_manager.metrics.snapshot()
                out_tokens[policy] = tok
                if policy == "none":
                    none_ttfts.append(ttft_ms)
                else:
                    lru_ttfts.append(ttft_ms)
                    reused = after["reused_tokens"] - before["reused_tokens"]
                    assert reused == shared_len, (shared_len, reused)
                    reused_tokens_list.append(reused)

            if out_tokens["none"] == out_tokens["lru"]:
                exact_matches += 1

        none_stats = summarize_ms(none_ttfts)
        lru_stats = summarize_ms(lru_ttfts)
        p50_reduction = (none_stats["p50_ms"] - lru_stats["p50_ms"]) / none_stats["p50_ms"] * 100.0
        results.append({
            "prompt_tokens": prompt_len,
            "shared_prefix_tokens": shared_len,
            "shared_ratio": round(shared_len / prompt_len, 4),
            "none": none_stats,
            "lru": lru_stats,
            "ttft_p50_reduction_pct": round(p50_reduction, 2),
            "avg_reused_tokens": statistics.mean(reused_tokens_list),
            "greedy_exact_match": f"{exact_matches}/{repeats}",
        })
        print(
            f"[Ratio {shared_len:4d}/{prompt_len}] "
            f"No-Cache P50={none_stats['p50_ms']:.2f}ms | "
            f"LRU P50={lru_stats['p50_ms']:.2f}ms ({p50_reduction:+.1f}%) | "
            f"Exact={exact_matches}/{repeats}"
        )

    return results


def build_tight_cache_trace(tokenizer, cycles: int = 6, prefix_blocks: int = 4):
    """
    构造紧 KV Cache 容量（默认 prefix_blocks=4 -> 1024-token 前缀 + 64-token 尾块 = 1088 tokens，
    单条长请求占 5 块；cache_blocks=14）的混合负载流：
    - Hot A / Hot B：各自拥有 prefix_blocks 个完整块共享系统提示词 + 64-token 唯一查询
    - Cold：prefix_blocks 个完整块一次性前缀 + 64-token 尾块
    - Short：64-token 短对话（占 1 个未哈希尾块）

    总容量 14 块 = Hot A(4) + Hot B(4) + 工作/冷缓冲(6)：
    - baseline (deque)：短请求释放的未哈希尾块排在队列后端，提前把 Hot A/B 挤出缓存；
    - fifo：Hot A/B 即使反复命中也不更新写入顺序，冷请求进来直接淘汰 Hot A/B 的根块，导致整条链失配；
    - lru：优先回收未哈希块 + 刷新 Hot A/B recency，冷请求只淘汰旧冷块，稳定保住双热点！
    """
    toks = [
        tokenizer.encode(w, add_special_tokens=False)[0]
        for w in ("系统", "助手", "代码", "推理", "分析", "性能", "测试", "优化", "数学", "对话")
    ]
    hot_a_prefix = []
    hot_b_prefix = []
    for b in range(prefix_blocks):
        hot_a_prefix.extend([toks[b % 2]] * 256)
        hot_b_prefix.extend([toks[2 + (b % 2)]] * 256)

    trace = []
    uid = 0

    def next_suffix(length: int) -> list[int]:
        nonlocal uid
        uid += 1
        # 前两个 token 唯一编码请求编号，避免尾块意外相同
        return [toks[uid % len(toks)], toks[(uid // len(toks)) % len(toks)]] + [toks[4]] * (length - 2)

    def make_cold(cold_idx: int) -> list[int]:
        # 每个冷请求每个块的第一个 token 均唯一，确保不命中历史缓存
        blocks = []
        for b in range(prefix_blocks):
            blocks.extend([100 + cold_idx * 16 + b] + [toks[5 + (b % 2)]] * 255)
        return blocks + next_suffix(64)

    # 预热两段热点前缀进入缓存（计入完整流）
    trace.append(("hot_a", hot_a_prefix + next_suffix(64)))
    trace.append(("hot_b", hot_b_prefix + next_suffix(64)))

    cold_counter = 0
    for _ in range(cycles):
        trace.append(("short", next_suffix(64)))
        trace.append(("hot_a", hot_a_prefix + next_suffix(64)))
        trace.append(("short", next_suffix(64)))
        trace.append(("hot_b", hot_b_prefix + next_suffix(64)))
        cold_counter += 1
        trace.append(("cold", make_cold(cold_counter)))
        trace.append(("hot_a", hot_a_prefix + next_suffix(64)))
        trace.append(("short", next_suffix(64)))
        trace.append(("hot_b", hot_b_prefix + next_suffix(64)))

    return trace


def run_policy_comparison(
    llm: LLM,
    cycles: int = 6,
    rounds: int = 3,
    cache_blocks: int = 14,
    prefix_blocks: int = 4,
):
    """
    实验 B：紧 KV Cache 容量下，在相同混合请求流上对比
    none / baseline / fifo / lru 四种策略的命中率、复用 Token、淘汰次数与 TTFT。
    """
    params = SamplingParams(temperature=1e-5, max_tokens=1, ignore_eos=True)
    trace = build_tight_cache_trace(llm.tokenizer, cycles=cycles, prefix_blocks=prefix_blocks)
    policies = ["none", "baseline", "fifo", "lru"]

    # 预热模型
    reset_scheduler(llm, "none")
    run_single_ttft(llm, trace[0][1], params, seed=1)

    per_policy_runs = {p: [] for p in policies}
    ref_outputs = None

    for r in range(rounds):
        # 每轮旋转策略顺序，消除先后热机偏差
        rotated = policies[r % len(policies):] + policies[:r % len(policies)]
        for policy in rotated:
            reset_scheduler(llm, policy, num_kvcache_blocks=cache_blocks)
            all_ttfts = []
            hot_ttfts = []
            outputs = []
            for idx, (kind, prompt) in enumerate(trace):
                ttft_ms, tok = run_single_ttft(llm, prompt, params, seed=5000 + idx)
                all_ttfts.append(ttft_ms)
                if kind in ("hot_a", "hot_b"):
                    hot_ttfts.append(ttft_ms)
                outputs.append(tok)

            if policy == "none" and ref_outputs is None:
                ref_outputs = outputs

            metrics = llm.scheduler.block_manager.metrics.snapshot()
            matches = sum(a == b for a, b in zip(outputs, ref_outputs)) if ref_outputs else len(outputs)
            per_policy_runs[policy].append({
                "round": r + 1,
                "all_ttft": summarize_ms(all_ttfts),
                "hot_ttft": summarize_ms(hot_ttfts),
                "total_prefill_ms": sum(all_ttfts),
                "metrics": metrics,
                "exact_matches": matches,
                "total_requests": len(trace),
            })

    summary = {}
    none_total_ms = statistics.mean(run["total_prefill_ms"] for run in per_policy_runs["none"])
    none_hot_p50 = statistics.mean(run["hot_ttft"]["p50_ms"] for run in per_policy_runs["none"])

    print(f"\n=== Tight Cache Policy Comparison (cache_blocks={cache_blocks}, requests={len(trace)}, rounds={rounds}) ===")
    for policy in policies:
        runs = per_policy_runs[policy]
        m = runs[0]["metrics"]  # 确定性流，每轮 cache metrics 完全一致
        hit_rate = (m["hit_count"] / m["lookup_count"] * 100.0) if m["lookup_count"] > 0 else 0.0
        all_p50 = statistics.mean(r["all_ttft"]["p50_ms"] for r in runs)
        all_p95 = statistics.mean(r["all_ttft"]["p95_ms"] for r in runs)
        hot_p50 = statistics.mean(r["hot_ttft"]["p50_ms"] for r in runs)
        hot_p95 = statistics.mean(r["hot_ttft"]["p95_ms"] for r in runs)
        total_ms = statistics.mean(r["total_prefill_ms"] for r in runs)
        exact = min(r["exact_matches"] for r in runs)

        summary[policy] = {
            "block_hit_rate_pct": round(hit_rate, 2),
            "lookup_count": m["lookup_count"],
            "hit_count": m["hit_count"],
            "reused_blocks": m["reused_blocks"],
            "reused_tokens": m["reused_tokens"],
            "eviction_count": m["eviction_count"],
            "all_ttft_p50_ms": round(all_p50, 2),
            "all_ttft_p95_ms": round(all_p95, 2),
            "hot_ttft_p50_ms": round(hot_p50, 2),
            "hot_ttft_p95_ms": round(hot_p95, 2),
            "total_prefill_ms": round(total_ms, 2),
            "total_prefill_reduction_vs_none_pct": round((none_total_ms - total_ms) / none_total_ms * 100.0, 2),
            "hot_p50_reduction_vs_none_pct": round((none_hot_p50 - hot_p50) / none_hot_p50 * 100.0, 2),
            "greedy_exact_match": f"{exact}/{len(trace)}",
            "rounds": runs,
        }
        print(
            f"  {policy:8s} | HitRate={hit_rate:5.1f}% | Reused={m['reused_tokens']:5d} tok | "
            f"Evict={m['eviction_count']:2d} | Hot P50={hot_p50:6.2f}ms | "
            f"All P95={all_p95:6.2f}ms | Total={total_ms:7.1f}ms | Match={exact}/{len(trace)}"
        )

    return {
        "cache_blocks": cache_blocks,
        "block_size": llm.model_runner.config.kvcache_block_size,
        "num_requests": len(trace),
        "rounds": rounds,
        "summary": summary,
    }


def collect_metadata(repo: str, model_path: str) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    with open(__file__, "rb") as f:
        script_hash = hashlib.sha256(f.read()).hexdigest()
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
    }
    for mod in ("triton", "flash_attn"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "installed")
        except Exception:
            env[mod] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "benchmark_script_sha256": script_hash,
        "environment": env,
        "model": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Prefix Cache 复用与淘汰策略（None / Baseline / FIFO / LRU）基准测试")
    parser.add_argument("--model", required=True, help="本地模型目录")
    parser.add_argument("--mode", choices=["pair", "eval"], default="eval")
    parser.add_argument("--pairs", type=int, default=3, help="pair 模式重复组数")
    parser.add_argument("--repeats", type=int, default=12, help="eval 模式共享比例扫描每档重复次数")
    parser.add_argument("--cycles", type=int, default=6, help="eval 模式紧缓存混合流循环次数")
    parser.add_argument("--rounds", type=int, default=3, help="eval 模式四策略轮换测量轮数")
    parser.add_argument("--cache-blocks", type=int, default=14, help="eval 模式紧缓存实验物理块上限")
    parser.add_argument("--prefix-blocks", type=int, default=4, help="eval 模式紧缓存实验长请求完整前缀块数")
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
    )

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report = collect_metadata(repo, args.model)

    if args.mode == "pair":
        for shared in (False, True):
            run_pair_case(llm, shared, 0)
        runs = []
        for pair in range(1, args.pairs + 1):
            order = (False, True) if pair % 2 else (True, False)
            for shared in order:
                res = run_pair_case(llm, shared, pair)
                res["pair"] = pair
                runs.append(res)
                print(
                    f"第 {pair} 组 {res['case']}：TTFT {res['ttft_ms']:.2f} ms，"
                    f"实际复用 {res['metrics']['reused_tokens']} Token"
                )
        report["mode"] = "pair"
        report["runs"] = runs
    else:
        print("=== Experiment A: Shared Prefix Ratio Sweep (1664 tokens, No-Cache vs LRU) ===")
        ratio_results = run_ratio_sweep(llm, repeats=args.repeats)
        policy_results = run_policy_comparison(
            llm,
            cycles=args.cycles,
            rounds=args.rounds,
            cache_blocks=args.cache_blocks,
            prefix_blocks=args.prefix_blocks,
        )
        report["mode"] = "eval"
        report["ratio_sweep"] = ratio_results
        report["policy_comparison"] = policy_results

    directory = os.path.join(repo, "results")
    os.makedirs(directory, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime(f"prefix_cache_{args.mode}_%Y%m%d_%H%M%S_%f.json")
    path = os.path.join(directory, filename)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{path}")


if __name__ == "__main__":
    main()
