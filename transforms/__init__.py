import torch
import torch.nn as nn


class Multiply(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp * self.val


class Divide(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp / self.val


class Interpolate(nn.Module):
    """Interpolate tensor values between target min/max"""
    def __init__(self, tgt_min: float, tgt_max: float, mask: torch.Tensor | None = None):
        super().__init__()
        self.tgt_min = tgt_min
        self.tgt_max = tgt_max
        self.mask = mask

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            mask = torch.ones_like(inp).bool()
        else:
            mask = self.mask

        src_min = inp[mask].min()
        src_max = inp[mask].max()

        out = inp

        a = (inp[mask] - src_min) / (src_max - src_min)
        out[mask] = self.tgt_min + a * (self.tgt_max - self.tgt_min)

        return out
