import torch
import torch.nn as nn
__all__ = ['Bifpn']
class Bifpn(nn.Module):
    def __init__(self, inc_list):
        super().__init__()
        self.layer_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        layer_weight = self.relu(self.layer_weight.clone())
        layer_weight = layer_weight / (torch.sum(layer_weight, dim=0))
        return torch.sum(torch.stack([layer_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)