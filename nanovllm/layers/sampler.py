import torch
from torch import nn


class Sampler(nn.Module):
    # 采样流程：Logits → Temperature Scaling → Softmax → 随机采样 → Token ID
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 1. 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 2. 将 logits 转换为概率分布
        probs = torch.softmax(logits, dim=-1)
        # 3. 利用指数分布进行随机采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens