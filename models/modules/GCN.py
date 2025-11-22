import math
from copy import copy


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch.nn import Parameter

from ..common import Conv

import torch
import pickle



__all__ = ['GCN5','Mix' ]


###### Atten ########
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x
########end of Atten #########


####### GCN Layers #######
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print("self.weight=",self.weight)
        input = input.to(self.weight.device)
        support = torch.matmul(input.to(torch.float32), self.weight.to(torch.float32))
        output = torch.matmul(adj, support.to(adj.dtype))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1

    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    #_adj = _adj + np.identity(num_classes, np.int)
    _adj = _adj + np.identity(num_classes, dtype=int)  # or np.int64

    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A.to(D.dtype), D).t(), D)
    return adj


class GCN5(nn.Module):
    def __init__(self, c1, c2, num_classes=4, t=0.3,
                  adj_file='new_adj_matrix.pkl',
                  inp_name='new_clip.pkl'):  # model
                 #adj_file='adjacency_info.pkl',
                 #inp_name='cls_embedding.pkl'):  # model
        # t=0.02
        super().__init__()
        # self.features = model
        self.linear_out_channel_list = [128, 256, 512]
        self.num_classes = num_classes
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        #self.gc1 = GraphConvolution(300, 1024)
        self.gc1 = GraphConvolution(512, 1024)
        self.gc2 = GraphConvolution(1024, c2)
        self.relu = nn.LeakyReLU(0.2)
        self.act = nn.SiLU()

        self.linear = nn.Linear(4, 1024)
        self.linear128 = nn.Linear(1024, self.linear_out_channel_list[0])
        self.linear256 = nn.Linear(1024, self.linear_out_channel_list[1])
        self.linear512 = nn.Linear(1024, self.linear_out_channel_list[2])
        self.sigmoid = nn.Sigmoid()
        self.se_layer128 = SqueezeExcitation(128)
        self.se_layer256 = SqueezeExcitation(256)
        self.se_layer512 = SqueezeExcitation(512)
        self.conv128 = nn.Conv2d(128, 64, 1)
        self.conv256 = nn.Conv2d(256, 128, 1)
        self.conv512 = nn.Conv2d(512, 256, 1)
        self.conv1024 = nn.Conv2d(1024, 512, 1)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

    # feature 就是CNN网络的输入 inp就是word embedding
    def forward(self, input_feature):
        feature_dim = input_feature.shape[1]
        feature = self.pooling(input_feature)  # feature.shape 4*2048*1*1
        feature = feature.view(feature.size(0), -1)
        # 2层的GCN网络
        # word embedding ##### num_class * c2
        inp = torch.tensor(self.inp)
        # print("inp shape:", inp.shape)  # 确保输出的形状为 (num_classes, 300)
        # exit()
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)  # 5*1024 ->
        x = self.relu(x)
        x = self.gc2(x, adj)  # 5*256  5*512
        ########
        x = x.transpose(0, 1)  # c2 * 5
        x = torch.matmul(feature.to(x.dtype), x)  # B * c2  c2 *5 == B * 5 (B * cls)
        x = self.linear(x)
        if feature_dim == 128:
            x = self.linear128(x)
            attention_weight = self.sigmoid(x).unsqueeze(-1).unsqueeze(-1)
            feature_info = input_feature * attention_weight
            feature_info = self.se_layer128(feature_info)
            feature_info = self.conv128(feature_info)
            input_feature = self.conv128(input_feature)
            # feature = self.conv256(torch.cat((input_feature, feature_info), dim=1))

        if feature_dim == 256:
            x = self.linear256(x)
            attention_weight = self.sigmoid(x).unsqueeze(-1).unsqueeze(-1)
            feature_info = input_feature * attention_weight
            feature_info = self.se_layer256(feature_info)
            feature_info = self.conv256(feature_info)
            input_feature = self.conv256(input_feature)
            # feature = self.conv512(torch.cat((input_feature, feature_info), dim=1))

        if feature_dim == 512:
            x = self.linear512(x)
            attention_weight = self.sigmoid(x).unsqueeze(-1).unsqueeze(-1)
            feature_info = input_feature * attention_weight
            feature_info = self.se_layer512(feature_info)
            feature_info = self.conv512(feature_info)
            input_feature = self.conv512(input_feature)
            # feature = self.conv1024(torch.cat((input_feature, feature_info), dim=1))

        feature = torch.cat((input_feature, feature_info), dim=1)

        return feature




###### end of GCN Layers #######



class Mix(nn.Module):
    def __init__(self, c1, c2):
        super(Mix, self).__init__()
        # self.m = 0.80
        self.FeatureBanlance = torch.nn.Parameter(torch.tensor(0.8), requires_grad=True)
        # w = torch.nn.Parameter(w, requires_grad=True)
        # self.w = w
        self.mix_block = nn.Sigmoid()
        self.cov = Conv(c1, c2)

    def forward(self, fea_list):
        mix_factor = self.mix_block(self.FeatureBanlance)
        out = self.cov(fea_list[0]) * mix_factor + fea_list[1] * (1 - mix_factor)
        return out

        
# class Mix(nn.Module):
#     def __init__(self, c1, c2):
#         super(Mix, self).__init__()
#         self.FeatureBanlance = torch.nn.Parameter(torch.tensor(0.8), requires_grad=True)
#         self.mix_block = nn.Sigmoid()
#         self.cov = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)  # 保持输入特征图尺寸

#     def forward(self, fea_list):
#         # feature mix
#         mix_factor = self.mix_block(self.FeatureBanlance)
#         fea1 = fea_list[0]  # 特征图1经过卷积操作
#         fea2 = fea_list[1]  # 特征图2直接使用
        
#         out = fea1 * mix_factor + fea2 * (1 - mix_factor)  # 线性组合
#         return out

################## start of restore decoder ###########################
