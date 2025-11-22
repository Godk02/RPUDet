import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import pickle
import numpy as np
import math


__all__ = ['EGCN' ]


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

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

class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale)
        return scale * x
class SimpleChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimpleChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


####### GCN Layers #######

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(1, 1, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        device = input.device  # 获取输入张量的设备（CPU 或 GPU）
        
        input = input.to(device, dtype=torch.float32)  # 确保输入张量是 float32
        self.weight = self.weight.to(device, dtype=torch.float32)  # 确保权重张量是 float32
        adj = adj.to(device, dtype=torch.float32)  # 确保邻接矩阵是 float32

        support = torch.matmul(input, self.weight)  # 计算支持
        output = torch.matmul(adj, support)  # 矩阵乘法

        if self.bias is not None:
            return output + self.bias
        return output



class EGCN(nn.Module):
    def __init__(self, c1, c2, num_classes=10, t=0.3, adj_file='visdrone_adj_matrix.pkl', inp_name='visdrone_clip.pkl'):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.gc1 = GraphConvolution(512, 1024)
        self.gc2 = GraphConvolution(1024, c2)
        self.relu = nn.LeakyReLU(0.2)
        self.act = nn.SiLU()

        self.linear_out_channel_list = [128, 256, 512]
        self.num_classes = num_classes
        self.linear = nn.Linear(10, 1024) ###################修改类别
        self.linear_layers = nn.ModuleList([
            nn.Linear(1024, 128),
            nn.Linear(1024, 256),
            nn.Linear(1024, 512)
        ])
        
        self.sigmoid = nn.Sigmoid()
        self.attention_layers = nn.ModuleList([
            SimpleChannelAttention(128),
            SimpleChannelAttention(256),
            SimpleChannelAttention(512)
        ])

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(512, 256, 1)
        ])

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.tensor(_adj).float())  # Ensure this is float32

        # Load the adjacency matrix and input features (ensure correct data type and device)
        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)

    def forward(self, input_feature):
        device = input_feature.device  # Get the device of the input feature

        # Ensure inp and adj are on the same device as input_feature
        inp = torch.tensor(self.inp).to(device, dtype=torch.float32)
        adj = gen_adj(self.A).to(device, dtype=torch.float32).detach()

        # Process input feature
        feature_dim = input_feature.shape[1]
        feature = self.pooling(input_feature).view(input_feature.size(0), -1)

        # Graph convolution and attention
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        # Matmul and linear layers
        x = x.transpose(0, 1)  # c2 * 5
        x = torch.matmul(feature.to(x.dtype), x)  # B * c2  c2 *5 == B * 5 (B * cls)
        x = self.linear(x)

        # Choose the correct linear layer based on feature dimension
        idx = [128, 256, 512].index(feature_dim)
        x = self.linear_layers[idx](x)

        # Attention mechanism
        attention_weight = self.sigmoid(x).unsqueeze(-1).unsqueeze(-1)
        feature_info = input_feature * attention_weight

        # Attention layers and convolution
        feature_info = self.attention_layers[idx](feature_info)
        feature_info = self.conv_layers[idx](feature_info)
        input_feature = self.conv_layers[idx](input_feature)

        return torch.cat((input_feature, feature_info), dim=1)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# #import clip
# from torch.nn.parameter import Parameter
# import numpy as np
# import pickle
# import math

# # GCN辅助函数
# def gen_A(num_classes, t, adj_file):
#     result = pickle.load(open(adj_file, 'rb'))
#     _adj = result['adj']
#     _nums = result['nums']
#     _nums = _nums[:, np.newaxis]
#     _adj = _adj / _nums
#     _adj[_adj < t] = 0
#     _adj[_adj >= t] = 1
#     _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
#     _adj = _adj + np.identity(num_classes, dtype=int)
#     return _adj

# def gen_adj(A):
#     D = torch.pow(A.sum(1).float(), -0.5)
#     D = torch.diag(D)
#     adj = torch.matmul(torch.matmul(A.to(D.dtype), D).t(), D)
#     return adj

# # GCN层
# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, bias=False):
#         super(GraphConvolution, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
#         self.bias = nn.Parameter(torch.Tensor(1, 1, out_features)) if bias else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):
#         device = input.device  # 获取输入张量的设备（CPU 或 GPU）
        
#         input = input.to(device, dtype=torch.float32)  # 确保输入张量是 float32
#         self.weight = self.weight.to(device, dtype=torch.float32)  # 确保权重张量是 float32
#         adj = adj.to(device, dtype=torch.float32)  # 确保邻接矩阵是 float32

#         support = torch.matmul(input, self.weight)  # 计算支持
#         output = torch.matmul(adj, support)  # 矩阵乘法

#         if self.bias is not None:
#             return output + self.bias
#         return output


# # 简单的通道注意力机制
# class SimpleChannelAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SimpleChannelAttention, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // 2),
#             nn.ReLU(),
#             nn.Linear(in_channels // 2, in_channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         scale = self.pool(x).view(b, c)
#         scale = self.fc(scale).view(b, c, 1, 1)
#         return x * scale


# class EGCN(nn.Module):
#     def __init__(self, c1, c2, num_classes=4, adj_file='new_adj_matrix.pkl', inp_name='new_clip.pkl'):
#         super(EGCN, self).__init__()

#         # 图像特征处理部分
#         self.pooling = nn.AdaptiveMaxPool2d((1, 1))  # 池化到 1x1
#         self.gc1 = GraphConvolution(512, 1024)
#         self.gc2 = GraphConvolution(1024, c2)
#         self.relu = nn.LeakyReLU(0.2)
#         self.act = nn.SiLU()

#         # 用于图像特征的全连接层
#         self.linear = nn.Linear(1024, 4)  # 输入 1024，输出 4（类别数量）
#         self.linear_layers = nn.ModuleList([
#             nn.Linear(1024, 128),
#             nn.Linear(1024, 256),
#             nn.Linear(1024, 512)
#         ])

#         # 加载邻接矩阵
#         _adj = gen_A(num_classes, 0.3, adj_file)
#         self.A = nn.Parameter(torch.tensor(_adj).float())  # Ensure this is float32

#         # 文本特征处理部分
#         with open(inp_name, 'rb') as f:
#             self.text_features = pickle.load(f)  # 加载类别文本特征
#         self.text_features = torch.tensor(self.text_features).float()  # 确保文本特征是float32

#         # 其它层
#         self.sigmoid = nn.Sigmoid()
#         self.attention_layers = nn.ModuleList([
#             SimpleChannelAttention(128),
#             SimpleChannelAttention(256),
#             SimpleChannelAttention(512)
#         ])

#         self.conv_layers = nn.ModuleList([
#             nn.Conv2d(128, 64, 1),
#             nn.Conv2d(256, 128, 1),
#             nn.Conv2d(512, 256, 1)
#         ])
        
#     def forward(self, input_feature):
#         device = input_feature.device  # 获取输入设备
        
#         # 图像特征处理
#         feature_dim = input_feature.shape[1]
#         feature = self.pooling(input_feature).view(input_feature.size(0), -1)  # shape should be [batch_size, channels]

#         # 获取图像特征和文本特征
#         inp = torch.tensor(self.text_features).to(device, dtype=torch.float32)  # 确保文本特征在相同设备上
#         adj = gen_adj(self.A).to(device, dtype=torch.float32).detach()

#         # 图像的图卷积操作
#         x = self.gc1(inp, adj)
#         x = self.relu(x)
#         x = self.gc2(x, adj)

#         # 处理文本特征的维度，确保它与图像特征的维度一致
#         batch_size = input_feature.size(0)
#         text_features = self.text_features[:batch_size, :].to(device)  # 将文本特征移动到与输入相同的设备

#         # 如果文本特征维度和图像特征维度不一致，可以做池化操作或者选取合适的文本特征
#         assert feature.size(1) == text_features.size(1), f"Feature and text features dimensions don't match: {feature.size(1)} vs {text_features.size(1)}"
        
#         # 将图像特征与文本特征拼接
#         x = torch.cat((feature, text_features), dim=-1)  # 拼接成更大的特征向量

#         # 检查拼接后的维度是否为 [batch_size, 1024]
#        # print(f"Shape after concatenation: {x.shape}")  # 输出拼接后的维度，应该是 [batch_size, 1024]

#         # 确保维度为 [batch_size, 1024] 以便进入全连接层
#         if x.shape[1] != 1024:
#             x = x.view(x.size(0), 1024)  # 调整维度，确保进入 linear 层

#         # 通过 feature_dim 选择正确的 linear 层
#         if feature_dim == 128:
#             idx = 0
#         elif feature_dim == 256:
#             idx = 1
#         elif feature_dim == 512:
#             idx = 2
#         else:
#             raise ValueError(f"Unsupported feature_dim: {feature_dim}")

#         # 选择对应的 linear 层
#         x = self.linear_layers[idx](x)

#         # 使用Sigmoid和注意力机制进行加权
#         attention_weight = self.sigmoid(x).unsqueeze(-1).unsqueeze(-1)
#         feature_info = input_feature * attention_weight

#         # 最后使用卷积层
#         feature_info = self.attention_layers[idx](feature_info)
#         feature_info = self.conv_layers[idx](feature_info)
#         input_feature = self.conv_layers[idx](input_feature)

#         return torch.cat((input_feature, feature_info), dim=1)

# import torch
# from torchviz import make_dot

# # 假设输入图像特征是 [batch_size, channels, height, width] 的张量
# input_feature = torch.randn(1, 512, 64, 64)  # Example: [1, 512, 64, 64] (batch_size, channels, height, width)

# # 创建 EGCN 模型实例
# model = EGCN(c1=512, c2=512, num_classes=10)

# # 获取模型的输出
# output = model(input_feature)

# # 使用 torchviz 绘制前向传播图
# dot = make_dot(output, params=dict(model.named_parameters()))
# dot.render("EGCN_forward_graph", format="png")






