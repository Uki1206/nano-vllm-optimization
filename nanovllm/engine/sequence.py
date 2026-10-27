from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


"""
Sequence有三重身份(三个不同的模块拿到相同的Sequence,但关注的字段不同):
1. 对于tonizer / 用户,是用户请求的载体,主要关注seq_id, token_ids, num_prompt_tokens, temperature
2. 对于Scheduler, 是调度的最小单位, 主要关注status, num_scheduled_tokens, is_prefill
3. 对于BlockManager, 是KV块的持有者, 主要关注block_table, num_cached_tokens
并且他们都要关注总长度num_tokens
"""

class Sequence:
    block_size = 256        # 类变量，由 LLMEngine 启动时根据 Config 改写
    counter = count()       # 自增 seq_id 来源

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # -- 用户请求载体 --
        self.seq_id = next(Sequence.counter)            # 全局唯一编号
        self.token_ids = copy(token_ids)                # 完整 token 序列（prompt + 已生成）
        self.last_token = token_ids[-1]                 # token_ids[-1]的缓存，decode路径会高频访问
        self.num_prompt_tokens = len(token_ids)         # 原始prompt长度，永不变
        self.num_tokens = len(self.token_ids)           # token_ids总长度
        self.temperature = sampling_params.temperature  # 温度
        self.max_tokens = sampling_params.max_tokens    # 单个请求允许生成的最大 Token 数；每轮调度预算由 max_num_batched_tokens 控制
        self.ignore_eos = sampling_params.ignore_eos    # 是否忽略eos继续生成

        # -- 调度最小单位 --
        self.status = SequenceStatus.WAITING
        self.num_scheduled_tokens = 0                   # 当前 step 要计算的 token 数；非 step 期间 = 0
        self.is_prefill = True                          # 当前是否是prefill阶段

        # -- KV块持有者 --
        self.block_table = []                           # 当前Sequence所占的物理block块号列表
        self.num_cached_tokens = 0                      # 已写入 KV Cache的token数

    # 其他“属性”
    @property
    def is_finished(self):              # 当前seq的decode是否已结束(达到max_token/生成eos)
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):   # 已生成的token数（不含prompt）
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):         # prompt的token
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):     # 生成的token
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):               # 当前需要多少block块
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):    # 最后一个block内实际用了多少slot
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 其他方法
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    def block(self, i):                # 取第 i 个物理块对应的 token 列表（用于 hashing）
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i+1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        当 Sequence 对象需要被 pickle、跨进程传输或者存储时，不要把整个对象的所有属性都打包，而是只把
        这些真正需要的数据打包出去
        data = pickle.dumps(seq)时会调用state = seq.__getstate__()，然后实际上序列化的是这个 state
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens, 
            self.num_prompt_tokens, 
            self.num_cached_tokens, 
            self.num_scheduled_tokens, 
            self.block_table, 
            last_state
        )

    def __setstate__(self, state):
        """
        __setstate__ 和 __getstate__ 成对使用
            __getstate__：Sequence → 打包成 tuple
            __setstate__：tuple → 恢复成 Sequence
        这里没有额外传一个 is_prefill 布尔值，而是通过最后一个字段是 list 还是单个整数，直接判断当前属于 Prefill 还是 Decode
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, self.last_state = state
        if isinstance(self.last_state, list):
            self.token_ids = self.last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = self.last_state

    

        