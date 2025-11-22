import torch
import torch.nn as nn
from .SPDConv import * 
from ..common import * 
__all__ = ['SPD_ADown']

# -------------------------SPD_ADown____________________
class SPD_ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = SPDConv(c1 // 2, self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 3, 1, 1, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)