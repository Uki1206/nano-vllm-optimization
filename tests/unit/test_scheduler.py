from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def test_prefill_first():
    # 构造 Scheduler 需要的最小配置
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=8,
        eos=-1,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
    )

    Sequence.block_size = config.kvcache_block_size
    scheduler = Scheduler(config)

    params = SamplingParams(
        temperature=0.6,
        max_tokens=3,
        ignore_eos=True,
    )

    # 两个模拟请求，列表中的数字代表输入 Token ID
    request_a = Sequence([1, 2, 3, 4], params)
    request_b = Sequence([5, 6, 7, 8], params)

    names = {
        request_a.seq_id: "A",
        request_b.seq_id: "B",
    }

    def show_step(step, seqs, is_prefill):
        phase = "Prefill" if is_prefill else "Decode"
        scheduled_names = [names[seq.seq_id] for seq in seqs]
        print(f"第 {step} 轮：{phase}，调度请求 {scheduled_names}")

    # 第 1 轮：先提交 A，并模拟它完成一次 Prefill
    scheduler.add(request_a)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [request_a]
    show_step(1, seqs, is_prefill)

    # 用模拟的 Token ID 10 更新请求状态，不运行模型
    scheduler.postprocess(seqs, [10], is_prefill)

    # 此时 A 已生成第一个 Token；新请求 B 到来
    scheduler.add(request_b)

    # 第 2 轮：观察 Scheduler 选择 A 的 Decode 还是 B 的 Prefill
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [request_b]
    assert request_a.num_completion_tokens == 1
    show_step(2, seqs, is_prefill)

    # 模拟 B 完成 Prefill
    scheduler.postprocess(seqs, [11], is_prefill)

    # 第 3 轮：观察两个请求的 Decode
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs == [request_a, request_b]
    show_step(3, seqs, is_prefill)


if __name__ == "__main__":
    test_prefill_first()