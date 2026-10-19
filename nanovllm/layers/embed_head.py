import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

class VocabParallelEmbedding(nn.Module):
    """
    词表并行embedding层
    """
    def __init__(
            self, 
            num_embeddings: int,  # 完整词表大小
            embedding_dim: int    # 每个词向量维度(Hidden States 的维度)
        ):
        super().__init__()
        # 1.获取 TP 相关信息
        self.tp_rank = dist.get_rank()           # 当前 GPU 在通信组中的编号，从 0 开始
        self.tp_size = dist.get_world_size()     # 参与 TP 的 GPU 数量   
        # 2.计算每张 GPU 分配多少 Token
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings        # 完整词表大小       
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size     # 每张 GPU 负责的词表大小
        # 3.计算当前 GPU 负责的 Token ID 范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank         # 当前 GPU 负责的第一个 Token ID（包含）
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition   # 当前 GPU 负责范围的结束位置（不包含）
        # 4.创建当前 GPU 的 Embedding 权重
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 5.绑定自定义权重加载函数
        self.weight.weight_loader = self.weight_loader # 给 self.weight 对象添加一个 weight_loader 属性，指向当前类的 weight_loader() 方法
                                                       # 作用：从完整的预训练 Embedding 权重中，加载当前 GPU 负责的那部分权重


    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        从完整的 Embedding 权重中，切出当前 GPU 负责的部分，并加载到当前 GPU 的 self.weight 中

        Args:
            param：当前 GPU 上创建的 Embedding 参数，也就是 self.weight。
            loaded_weight：从模型 checkpoint 加载的完整 Embedding 权重。
        """
        param_data = param.data  # 获取当前 GPU 的参数数据
        shard_size = param_data.size(0)     # 当前 GPU 负责的词表大小
        start_idx = self.tp_rank * shard_size   # 计算当前 GPU 的起始索引
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)  # 从完整权重中切出当前 GPU 的部分
        param_data.copy_(loaded_weight)     # 将切分后的权重复制到模型参数


    def forward(self, x: torch.Tensor):
        """
        每张 GPU 只查询自己负责的 Token Embedding，不属于自己的 Token 用 0 填充，
        最后通过 all_reduce(SUM) 将各 GPU 的结果相加，得到完整的 Embedding 输出。
        注：每张GPU都会收到相同的x(Token序列)，但只有自己负责的那部分的Embedding权重在当前的GPU上
        """
        # 1.判断当前 Token 是否属于本 GPU
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) #布尔类型的 mask： Tue：当前 Token 属于本 GPU； False：当前 Token 不属于本 GPU。
            # 2.将全局 Token ID 转换为局部 Token ID
            x = mask * (x - self.vocab_start_idx) # 这里乘以 mask 的目的，是保证所有局部索引合法
            # 举例： x = [100,6000,200], GPU0: start_idx = 0 , GPU1: start_idx = 5000
            # 计算之后 GPU0的x: [True,False,True] * [100, 6000, 200] -> [100, 0 ,200]
            #         GPU1的x: [False,True,False] * [-4900, 1000, -4800] -> [0, 1000, 0]
        
        # 3.执行embedding查表，这一步相当于y = self.weight[x]
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 4.将不属于当前 GPU 的 Embedding 置零,上面各GPU的x里还有0，导致会去查weight[0],这里再用mask将这部分weight[0]置零,用了广播机制
            y = mask.unsqueeze(1) * y
            # 5.通过 all_reduce 合并结果:将所有 GPU 对应位置的 Tensor 元素相加，再将求和结果同步到每张 GPU 上
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    词表并行的语言模型输出层（LM Head），它负责将 Transformer 
    最后一层的 Hidden States 映射为词表上的 Logits，用于预测下一个 Token
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,  # 是否使用偏置
    ):
        assert not bias  # 当前实现不支持带偏置的 LM Head
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        词表并行 LM Head：将 Hidden States 映射为 Logits。
        每张 GPU 只计算自己负责的词表部分，最后将所有 Logits 收集到 Rank 0 并拼接。

        Args:
            x: Transformer 最后一层输出的 Hidden States
                Prefill: [total_tokens, hidden_size]
                Decode:  [batch_size, hidden_size]

        Returns:
            logits: [batch_size, vocab_size]
                    TP > 1 时仅 Rank 0 返回完整 Logits，其他 Rank 返回 None。
        """
        # 1.获取当前推理上下文
        context = get_context()

        # 2.Prefill阶段，只保留每个请求最后一个Token的Hidden State
        # 在自回归生成中，我们只需要利用每个请求最后一个 Token 的 Hidden State，预测下一个 Token
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1  # 计算每个请求最后一个token的索引
            x = x[last_indices].contiguous()    # x[last_indices]取出x中last_indices对应的行(可以理解为每个请求的最后一个token对应的行)，.contiguous() 用于确保选取后的 Tensor 具有连续的内存布局，方便后续矩阵运算
                                                # 这样prefill阶段的x就变成了[batch_size,vocab_size]
        
        # 3.计算局部logits，logits = X*WT
        logits = F.linear(x, self.weight)
        
        # 4.在 GPU 0 上创建接收 Logits 的列表（缓冲区)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 5.使用 dist.gather() 收集所有 GPU 的 Logits
            # dist.gather() 会把所有 GPU 的局部 Logits 收集到 GPU 0。因此只有 GPU 0 需要接收缓冲区。
            dist.gather(logits, all_logits, 0)
            # 6.拼接所有 GPU 的 Logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
                # logits的维度:[batch_size,vocab_size],这里是按最后一维拼接，即把所有GPU分别处理的词表拼起来        
        return logits
    