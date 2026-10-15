import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    实现 SwiGLU 中的激活与逐元素乘法部分
    output = SiLU(gate) * up

    注意：这里只实现SwiGLU的核心计算，不包含前面的Linear投影
    """
    @torch.compile  # 使用Pytorch编译优化，融合算子，减少Kernel Launch和中间显存读写
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
