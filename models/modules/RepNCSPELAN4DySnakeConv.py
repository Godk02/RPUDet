import torch
import torch.nn as nn
from .DySnakeConv import * 
from ..common import * 
__all__ = ['RepNCSPELAN4DySnakeConv']


# -------------------------RepNCSPELAN4DySnakeConv____________________
class RepNBottleneck_DySnakeConv(RepNBottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], s=1, g=g)
        self.add = shortcut and c1 == c2


class RepNCSP_DySnakeConv(RepNCSP):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DySnakeConv(c1, c_)
        self.cv2 = DySnakeConv(c1, c_)
        self.cv3 = DySnakeConv(2 * c_, c2)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4DySnakeConv(RepNCSPELAN4):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, c3, c4, c5)
        self.cv1 = Conv(c1, c3, k=1, s=1)
        self.cv2 = nn.Sequential(RepNCSP_DySnakeConv(c3 // 2, c4, c5), DySnakeConv(c4, c4, 3))
        self.cv3 = nn.Sequential(RepNCSP_DySnakeConv(c4, c4, c5), DySnakeConv(c4, c4, 3))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)