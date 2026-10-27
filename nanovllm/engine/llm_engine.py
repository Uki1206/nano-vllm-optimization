import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """
        初始化整个推理引擎
        """
        config_fields = {field.name for field in fields(Config)}  # 筛选Config支持的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}  # 过滤kwargs
        config = Config(model, **config_kwargs)  #真正创建配置
        Sequence.block_size = config.kvcache_block_size # Sequence类变量，所有Seqence公用的block_size
        
        self.ps = []  # 子进程
        self.events = [] # 进程间通信使用的Event
        ctx = mp.get_context("spawn")  # 指定python多进程使用"spawn"模式：创建全新的pythob进程

        # 创建子进程及其event,从1开始(0是主进程)
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events) # 创建rank0 ModelRunner
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        atexit.register(self.exit) #注册退出逻辑：python程序关闭时自动调用self.exit()


    def exit(self):
        """
        结束整个推理引擎
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()  # join:等待这个子进程真正退出

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        调用Scheduler的add方法，加入一个请求
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)  #文本转token
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)


    def step(self):
        """
        ！核心！ 一次推理迭代
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # prefill：本轮要处理的token数 = scheduled_token数之和
        # decode: 本轮要处理的token数 = 请求的数量,用负数区分prefill和decode

        # 所有进程执行一次推理
        token_ids = self.model_runner.call("run", seqs, is_prefill) 

        # 更新请求状态，判断是否结束
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 只返回已经完成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens


    def is_finished(self):
        return self.scheduler.is_finished()


    def generate(
            self,
            prompts: list[str] | list[list[int]],
            sampling_params: SamplingParams | list[SamplingParams],
            use_tqdm: bool = True
    ) -> list[str]:
        """
        完整生成流程（包含多次step）
        """
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            #sampling_params如果是列表：各请求用各自的参数；不是列表：简单让所有请求共用一个参数

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)  #添加请求

        outputs = {} #接收输出

        prefill_throughput = decode_throughput = 0. # 初始化吞吐量

        while not self.is_finished():
            t = perf_counter() # 记录开始时间

            # 进行一次推理(prefill/decode)
            output, num_tokens = self.step()

            # 计算吞吐量：num_tokens > 0说明是prefill, < 0说明是decode阶段
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)

            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 收集已完成请求(注意step中只返回已完成的请求)
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids  
                pbar.update(1)

        pbar.close()
        # 按seq_id恢复顺序，outputs由dict变为下标序的list
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # token转文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs



