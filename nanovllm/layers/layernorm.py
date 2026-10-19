import torch
from torch import nn

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))


    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准 RMSNorm
        """
        orig_dtype = x.dtype    
        x = x.float()           # FP16 -> FP32，中间计算保持精度
        var = x.pow(2).mean(dim=-1, keepdim=True)  # 计算均方值
        x.mul(torch.rsqrt(var + self.eps))  # x * (var + eps)的平方根倒数
        x = x.to(orig_dtype).mul_(self.weight)  # FP32 -> FP16,恢复原精度，并乘权重（可学习参数）
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合 Residual Add + RMSNorm
        h = x + Residual
        y = RMSNorm(h)
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float()) # x 和 residual 都转换为 FP32，然后相加,对应 h = x + Residual
        residual = x.to(orig_dtype) # 新的残差（h），供后续层使用
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None,) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

    