import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型，TP通信
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size = self.world_size, rank = rank)
        
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()


    def exit(self):
        """
        清理资源
        """
        if self.world_size > 1:
            self.shm.close()    # 先关闭共享内存
            dist.barrier()      # 所有进程同步
            if self.rank == 0:
                self.shm.unlink()  # rank0真正删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 释放CUDA Graph
        torch.cuda.synchronize()  # 确保GPU工作结束
        dist.destroy_process_group()  # 销毁NCCL通信组

    def loop(self):
        """
        TP从卡一直等待rank0的指令，该方法是rank>0的进程使用
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取rank0的指令，只允许TP从卡调用
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # wait:如果Event == False,阻塞等待
        n = int.from_bytes(self.shm.buf[0:4], "little") # 规定shm前4个字节存后面数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) # 取出当前数据(方法名+该方法的参数)
        self.event.clear() # Event恢复False
        return method_name, args


    def write_shm(self, method_name, *args):
        """
        rank0写共享内存，给其他GPU发命令
        """
        assert self.world_size > 1 and self.rank == 0
        # 以下为rank0 写共享内存的过程，共享内存结构为 数据长度n（前4字节） + data（随后n字节）
        data = pickle.dump([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data  
        # 以下为rank0 通知其他GPU启动的过程
        for event in self.event:
            event.set()   # set() : 把Event设置为True,从而唤醒在wait等待的进程

    def call(self, method_name, *args):
        """
        所有方法的统一调用接口,自动调用method_name(*args),和一般的调用不同的是，这样运行会让所有线程一起工作
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        跑一次假请求，测模型(Prefill)的峰值显存
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 构造最大seqs
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(self.config.max_num_seqs, max_num_batched_tokens // seq_len)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()


    def allocate_kv_cache(self):
        """
        计算能放多少kv block
        """
        config = self.config
        hf_config = config.hf_config
        
        free, total =  torch.cuda.mem_get_info()
        used = total - free   # 得到当前显存使用量

        peak = torch.cuda.memory_stats()["allocate_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocate_bytes.all.current"]
        # warmup_model后，peak - current 相当于模型运行时额外需要的最大显存

        # 计算每个block占的显存大小
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_ayyention_heads)
        per_token_bytes = 2 * hf_config.num_hidden_layers * num_kv_heads * head_dim * hf_config.dtype.itemsize
        block_bytes = per_token_bytes * self.block_size

        # 计算kv block的数量
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - (peak - current)) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 真正分配kv cache
        self.kv_cache = torch.empty(2,
                                    hf_config.num_hidden_layers, 
                                    config.num_kvcache_blocks, 
                                    self.block_size,
                                    num_kv_heads, 
                                    head_dim)

        # 把每一层attention的k_cache/v_cache指向这块统一的kv_cache
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs:list[Sequence]):
        """
        把各seq的block_table拼成Tensor,并把数据传到GPU
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table))
                        for seq in seqs]  # 长度不足max_len的用-1对齐
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables


    def prepare_prefill(self, seqs:list[Sequence]):
        """
        将多个 Sequence 中本轮需要执行 Prefill 的 token 整理成批量输入，
        并准备 Attention 计算和 KV Cache 写入所需要的元数据。
        """
        input_ids = []          # 本轮 Prefill 需要计算的所有 token ID（多个 Sequence 拼接成一维数组）
        positions = []          # 每个 token 在原始 Sequence 中的位置，用于 RoPE 位置编码
        cu_seqlen_q = []        # 各 Sequence 的 Q 累计长度，用于区分不同 Sequence 的 Q（只包含本轮计算的 token）
        cu_seqlen_k = []        # 各 Sequence 的 K 累计长度，用于区分不同 Sequence 的 K（包含历史缓存 + 本轮计算的 token）
        max_seqlen_q = 0        # 本批次所有 Sequence 中最大的 Q 长度
        max_seqlen_k = 0        # 本批次所有 Sequence 中最大的 K 长度（包含历史 KV）
        slot_mapping = []       # 本轮各 token 对应的 KV Cache 物理 slot 索引，用于将新计算的 K/V 写入对应位置
        block_tables = None     # 逻辑 Block 到物理 Block 的映射表，用于 Attention 定位并读取历史 KV Cache

        for seq in seqs: 
            start = seq.num_cached_tokens  # 前面多少token已经存在kvcache，不需要重新计算
            seqlen_q = seq.num_scheduled_tokens # 本轮真正要计算多少个新token
            end = start + seqlen_q
            seqlen_k = end
            # 注意：Q只包含新计算的token，但K/V包含缓存的token+新token，因此Q和K/V长度不同
            
            # 把这轮要计算的token及位置保存
            input_ids.extend(seq[start:end])
            positions.extend(range(start,end))

            cu_seqlen_q.append(cu_seqlen_q[-1] + seqlen_q)
            cu_seqlen_k.append(cu_seqlen_k[-1] + seqlen_k)
                # 假如cu_seq_len = [0,3,8,10],表示seq1:[0:3] seq2:[3:8] seq3:[8:10]
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            if not seq.block_table:
                continue

            # 接下来处理slot_mapping:当前token的kv应该写到哪里
            start_block = start // self.block_size
            end_block = (end + self.block_size -1) // self.block_size
            for i in range(start_block, end_block):
                # block_table[i]是物理block块号，即Block粒度下在全局kv cache的起始位置， 乘block_size后相当于token粒度下在全局kv cache的起始位置
                slot_start = seq.block_table[i] * self.block_size
                # 若当前块是首块，处理首块的块内偏移
                if i == start_block:
                    slot_start += start % self.block_size
                # 若当前块是中间块，则中间的块一定写满
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:  # 若当前块是尾块，处理尾块的块内偏移
                    slot_end = seq.block_table[i] * self.block_size + (end - i * self.block_size)

                # 记录当前块的start,end
                slot_mapping.extend(range(slot_start, slot_end))

        if cu_seqlen_k[-1] > cu_seqlen_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        # cu_seqlens_q[-1]：本批次所有 Sequence 本轮需要计算的 token 总数。
        # cu_seqlens_k[-1]：本批次所有 Sequence 参与 Attention 的 K token 总数，包括已经缓存的历史 token。
        # 后者 > 前者，表明至少有一个seq存在已经缓存的历史KV
        # slot_mapping 解决新 KV 写到哪里的问题，block_tables 解决历史 KV 从哪里读取的问题

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlen_q = torch.tensor(cu_seqlen_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlen_k = torch.tensor(cu_seqlen_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlen_q, cu_seqlen_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions


    def prepare_decode(self, seqs:list[Sequence]):
        """
        准备Decode输入
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq)) # 虽然Q只有最后一个token，但它计算Attention时需要知道整个长度(取KV)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions


    def prepare_sample(self, seqs:list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures


    @torch.inference_mode()  #推理模式，禁用梯度计算
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        真正执行transformer，有普通pytorch和CUDA Graph两条执行路径
        """
        # 1.普通pytorch执行，条件:prefill阶段 / enforce_eager参数为True / Decode 阶段的 Batch Size 超过 51
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        # 2.CUDA Graph执行，条件：Decode阶段 且 enforce_eager=False 且 Decode 阶段的 Batch Size <= 512
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graph[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])


    def run(self, seqs:list[Sequence], is_prefill:bool) -> list[int]:
        """
        ModelRunner的核心执行入口，把之前的方法串起来
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # 从logits采样得到新token
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context() # 清空这一轮Attention Context
        return token_ids


    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        提前录制CUDA Graph,用于优化 Decode 阶段的推理性能
        """
        # 先准备一批固定大小的Tensor
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16,max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

