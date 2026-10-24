import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

def divide(numerator, denominator):
    """
    确保两个整数能够整除，然后返回整除结果
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    Linear 层的基类：
    负责初始化权重、偏置和 Tensor Parallel (TP) 相关信息。
    具体的 forward 计算由子类实现。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None):
        super().__init__()
        # 1. 初始化 Tensor Parallel 相关信息
        self.tp_dim = tp_dim                     # 权重的 TP 切分维度
        self.tp_rank = dist.get_rank()           # 当前 GPU 在分布式通信组中的 Rank
        self.tp_size = dist.get_world_size()     # TP 通信组中的 GPU 数量
        # 2. 初始化权重矩阵
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader #  # 给权重绑定自定义加载函数
        # 3. 初始化 Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))  # Bias 的大小等于当前 Linear 层的输出维度
            self.bias.weight_loader = self.weight_loader        # 给 Bias 绑定自定义权重加载函数
        else:
            self.register_parameter("bias", None)               # 不使用 Bias 时，将其注册为 None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LinearBase 是一个基类，只负责定义 Linear 层的公共属性和初始化逻辑，不负责实现具体的前向计算。
        NotImplementedError 表示：这个方法必须由子类实现，否则调用时就会报错

        不同 Linear 子类具有不同的计算及通信逻辑：
        1. ReplicatedLinear：普通线性计算，不需要 TP 通信
        2. ColumnParallelLinear：按输出维度切分权重，可能需要 All-Gather
        3. RowParallelLinear：按输入维度切分权重，通常需要 All-Reduce
        """
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    非切分线性层（Replicated Linear）:在 Tensor Parallel（TP）中，每张 GPU 都保存一份完整的权重矩阵，
    各自执行普通线性计算，不需要进行 TP 通信。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        param：当前 GPU 上需要加载权重的 Parameter。
        loaded_weight：从模型文件中读取的预训练权重。
        """
        param.data.copy_(loaded_weight)   # 将 loaded_weight 的数值复制到 param 中,不切分

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行线性变换linear：Y = X * WT + b
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层（Column Parallel Linear Layer）。
    将权重矩阵 W 按“列”（即输出特征维度 output_size）进行切分，分布到不同的 GPU 上。
    每个 GPU 接收完整的输入 X，独立计算自己负责的那部分输出特征 Y_i = X * W_i。  
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data                         # 获取当前 GPU 上已分配好的参数底层张量（形状已经是切分后的大小）
        shard_size = param_data.size(self.tp_dim)       # 获取当前分片在切分维度（self.tp_dim，即 dim=0）上的大小（即 output_size // tp_size）
        start_idx = self.tp_rank * shard_size           # 计算当前 GPU 在完整权重中的起始截取索引
        
        # 使用 narrow 方法在完整权重 loaded_weight 的 self.tp_dim 维度上，
        # 从 start_idx 开始截取长度为 shard_size 的连续切片
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)                 # 将截取出来的权重分片原位拷贝（in-place copy）到当前模型的参数中


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# 补充：为什么叫“列并行”却是切分第 0 维（tp_dim = 0）？
# 在数学公式 Y = XW + b 中，W 的形状是 [input_size, output_size]，切分输出维度就是按列切分。
# 但在 PyTorch 的 F.linear(x, weight, bias) 实现中，计算公式是 Y = xW^T + b，
# 因此 self.weight 的存储形状是 [output_size, input_size]。所以在代码底层切片时，操作的是第 0 维





class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并的列并行线性层。
    常用于将多个输入相同的线性层（例如 MLP 中的 gate_proj 和 up_proj）融合为一个大线性层，
    减少 CUDA Kernel 启动开销并提升矩阵乘法的计算效率。
    """

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        # output_sizes 是一个列表，记录了被合并的各个原始线性层的输出维度。
        # 例如合并 gate_proj 和 up_proj 时，output_sizes = [intermediate_size, intermediate_size]
        self.output_sizes = output_sizes
        # 此时当前 GPU 分配到的参数 param 形状为: [sum(output_sizes) // tp_size, input_size]
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        权重加载函数：由于预训练模型权重中各个子层（如 gate 和 up）是分开保存的，
        因此加载器每次只传入其中一个子层的完整权重 loaded_weight，并用 loaded_shard_id 标识它是第几个子层。
        
        参数:
            param: 当前 GPU 上的合并权重参数，形状为 [sum(output_sizes) // tp_size, input_size]
            loaded_weight: 从磁盘读取的某一个子层的完整预训练权重，形状为 [output_sizes[loaded_shard_id], input_size]
            loaded_shard_id: 当前加载的是第几个子层（例如 0 代表 gate_proj，1 代表 up_proj）
        """
        param_data = param.data
        # 计算当前子层在当前 GPU 合并参数 (param_data) 中的起始偏移量 (shard_offset)
        # 把排在当前子层前面的所有子层大小加起来，再除以 tp_size，就是前面子层在当前 GPU 上占用的空间大小
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size 
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size         # 当前子层分片在当前 GPU 上占用的大小 (shard_size)
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)   # 在当前 GPU 的合并权重张量上，“框”出属于当前子层的那一块内存区域（视图切片，不复制内存）
        # 将磁盘读入的完整子层权重 loaded_weight 沿着 tp_dim 均分成 tp_size 份，
        # 并取出属于当前 GPU (self.tp_rank) 的那一份
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        param_data.copy_(loaded_weight) #将切分好的子层权重拷贝到刚刚框出来的内存区域中



class QKVParallelLinear(ColumnParallelLinear):
    """
    将输入 Hidden States 投影为 Q、K、V，使用列并行，不同 GPU 负责不同 Head。
    """
    def __init__(
            self, 
            hidden_size: int,                       #模型隐藏层维度,d_model
            head_size: int,                         #单个注意力头的维度,即d_k或head_dim
            total_num_heads: int,                   #全局 Query (Q) 的总头数
            total_num_kv_heads: int | None = None,  #全局 Key (K) 和 Value (V) 的总头数,用于支持 GQA（分组查询注意力） 或 MQA
            bias: bool = False
        ):
        tp_size = dist.get_world_size()
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)       # 当前单张 GPU 负责的 Q 头数量
        self.num_kv_heads = divide(total_num_kv_heads, tp_size) # 当前单张 GPU 负责的 K 头（以及 V 头）数量
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size   # 合并后 Q、K、V 的全局总输出维度
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        参数说明:
        - param: 当前 GPU 上合并后的 QKV 权重参数，形状为 [(num_heads + 2 * num_kv_heads) * head_size, hidden_size]
        - loaded_weight: 从硬盘加载的完整单一部分权重（完整的 W_q、完整的 W_k 或完整的 W_v）
        - loaded_shard_id: 字符串标识，指明当前传入的 loaded_weight 是 "q"、"k" 还是 "v"
        """
        param_data = param.data
        assert loaded_shard_id in ["q","k","v"]

        # 第一步：计算当前加载的部分（q、k 或 v）在当前 GPU 的 param_data 中的【大小 (shard_size)】和【起始偏移量 (shard_offset)】
        # 注意：在当前 GPU 的合并矩阵中，排列顺序固定为 [Q分片, K分片, V分片]
        if loaded_shard_id == "q":
            # Q 分片排在最前面，所以偏移量是 0，大小是当前 GPU 负责的 Q 头数 × 单头维度
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            # K 分片紧跟在 Q 分片后面，所以偏移量等于 Q 分片的大小，大小是当前 GPU 负责的 KV 头数 × 单头维度
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            # V 分片排在最后（跳过前面的 Q 分片和 K 分片），所以偏移量是 Q 分片大小 + K 分片大小
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        # 第二步：在当前 GPU 的合并权重 param_data 上，“框”出对应区域（narrow 是视图操作，不拷贝内存）
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 第三步：将硬盘读入的完整权重（完整的 Q、K 或 V）沿着第 0 维（self.tp_dim）均分成 tp_size 份，
        # 并取出属于当前 GPU（self.tp_rank）的那一份
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        # 第四步：将切好的权重分片原位拷贝（in-place copy）到刚刚框出来的内存区域中
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层（Row Parallel Linear Layer）。
    将权重矩阵 W 按“行”（即输入特征维度 input_size）进行切分。
    它接收的输入 X 也是已经在输入维度上切分好的局部张量（通常直接来自上一层 ColumnParallelLinear 的输出）。
    每个 GPU 计算出局部结果后，通过 All-Reduce 跨卡求和得到最终完整输出。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False,):
        tp_size = dist.get_world_size()
        # 调用父类 LinearBase 初始化：
        # 1. divide(input_size, tp_size): 输入维度按 GPU 数量均分！
        # 2. output_size: 输出维度保持完整，不做切分！
        # 3. bias: 是否包含偏置。注意：因为输出维度是完整的，所以 bias 的形状也是完整的 [output_size]。
        # 4. 1: 指定切分维度 tp_dim = 1。
        #    PyTorch 的 weight 形状为 [output_size, input_size]，切分 input_size 对应的是第 1 维（dim=1）。
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # 【关键特判】如果是加载偏置项 bias（一维张量，形状为 [output_size]）：
        # 因为 RowParallelLinear 的输出维度没有切分，所以 bias 也不需要切分，直接完整拷贝即可。
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        # 以下是针对二维权重矩阵 weight 的标准切分套路（沿着 self.tp_dim = 1 切分）：
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算局部矩阵乘法 + 跨卡 All-Reduce 求和。
        输入 x 的形状: [batch_size, seq_len, input_size // tp_size]
        输出 y 的形状: [batch_size, seq_len, output_size]
        """
        # 1. 计算当前 GPU 的局部线性变换：Y_i = X_i * W_i^T
        # 【极其关键的细节】：self.bias if self.tp_rank == 0 else None
        # 只有 0 号 GPU (tp_rank == 0) 在计算时加上了 bias，其他 GPU 传的是 None（不加 bias）！
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 2. 如果开启了多卡张量并行，调用 all_reduce 将所有 GPU 算出的局部结果 y 按元素相加（Sum），
        # 并将相加后的完整结果同步回每一张 GPU。
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y