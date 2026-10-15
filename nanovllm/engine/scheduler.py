from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 每个step最多调度多少个seq
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 每个step的token最大值
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        完成一次调度，准备好scheduled_seqs 和 is_prefill, 默认Prefill优先
        """
        scheduled_seqs = []      # 本次step调度的seqs
        num_batched_tokens = 0   # 本次step调度seqs的总token数

        # ---1.prefill---
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens # 剩余token budget
            if remaining == 0:
                break
            if not seq.block_table:  
                # num_cached_blocks：当前这个请求有多少个完整 block 的前缀已经命中 Prefix Cache，可以直接复用，不需要重新计算。
                num_cached_blocks = self.block_manager.can_allocate(seq)  
                if num_cached_blocks == -1: # 如果不能分配就退出 
                    break
                # 真正需要计算的token数（没把remaining限制考虑在内）：去掉前缀命中的部分
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                
            else:  # 已经分配过kvcache/block,则seq的num_cached_tokens中记录了已经算过的token数
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if remaining < num_tokens and scheduled_seqs:
                break
            # 只有本轮第一个 prefill 请求允许被切成 chunk；如果前面已经调度了别的请求，当前请求放不下就直接留到下一轮。

            if not seq.block_table: # 这里才要真正分配
                # 首次 Prefill：为整个 Prompt 预分配所有需要的 Block
                # 后续 Chunked Prefill 直接复用，不需要再次分配，因此这里只有首次才分配
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining) # 本轮真正要计算的token数
            num_batched_tokens += seq.num_scheduled_tokens

            # prfill完成判断：已经计算过的token数 + 本轮要计算的token数 = 请求总长度
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:  # 如果调度队列不为空就直接返回，因此是prefill优先
            return scheduled_seqs, True


        # ---2.decode---
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()   # 从 running 队列左侧取出seq
            # 接下来要判断KV cache能否追加一个token
            while not self.block_manager.can_append(seq):  # 不能追加一个token，就要抢占
                if self.running:   # 若 decode 的队列里还有其他seq，优先抢占其他正在decode阶段的seq
                    self.preempt(self.running.pop())
                else:  # 否则抢占自己已有的kvcache空间
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
            
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self,seq: Sequence):
        """
        Decode阶段不够分配时的抢占规则,传入的是被抢占的seq
        被抢占的seq: 重新进入waiting队列，等待重新prifill，释放其所有物理block
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)   # 补偿：被抢占的seq到waiting的队头，方便下次立即prefill
        

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        每次step()后的后处理：更新 KV Cache 状态(更新hash,num_cached_tokens,num_scheduled_tokens)、
        保存新生成的 Token、判断请求是否结束，并释放已完成请求的资源
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)       # 更新hash_to_block_id
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # 如果当前seq是prefill阶段还没处理完，下一轮还要继续处理：先跳过当前seq
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 如果当前seq是Decode阶段，加入一个新token(注意：此处在prefill完成那一刻也会执行)
            seq.append_token(token_id)
                # decode结束：生成eos或达到允许的最大长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)







