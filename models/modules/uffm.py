# import inspect
# import math

# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F

# from torch.nn import BatchNorm2d as _BatchNorm
# from torch.nn import InstanceNorm2d as _InstanceNorm
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_



# class Mlp(nn.Module):

#     def __init__(self,
#                  in_features,
#                  hidden_features=None,
#                  out_features=None,
#                  act_layer=nn.GELU,
#                  drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)

#         x = self.dwconv(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)

#         return x



# # ----------------------------pam_cam------------------------------


# class PAM_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim

#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x):
#         m_batchsize, C, height, width = x.size()   # x:[B,C,H,W]
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)   # [B,C,H,W] -> [B,C///8 , H,W] -> [B, C//8 , H*W] -> [B , H*W ,C//8]
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)    # [B,C,H,W] -> [B,C///8 , H,W] -> [B, C//8 , H*W]

#         energy = torch.bmm(proj_query, proj_key)   # 使用 PyTorch 中的 torch.bmm() 函数执行了批量矩阵乘法（batch matrix multiplication）操作。
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma * out + x
#         return out


# class CAM_Module(nn.Module):

#     def __init__(self, in_dim):
#         super(CAM_Module, self).__init__()
#         self.chanel_in = in_dim
        
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax  = nn.Softmax(dim=-1)
#     def forward(self,x):
#         m_batchsize, C, height, width = x.size()   #  x:[B,C,H,W]  
#         proj_query = x.view(m_batchsize, C, -1)    #  x:[B,C,H,W]  -> [B,C,H*W]
#         proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)   #  [B,C,H,W]  -> [B,C,H*W] -> [B , H*W ,C ]
       
#         energy = torch.bmm(proj_query, proj_key)  # [B,C,C]
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # 这句代码我没看懂
#         attention = self.softmax(energy_new)
#         proj_value = x.view(m_batchsize, C, -1)

#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma * out + x
#         return out


# class basicBlock_dual(nn.Module):
#     def __init__(self,
#                  in_channels, out_channels):
#         super(basicBlock_dual, self).__init__()

#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#         )
#     def forward(self, x):
#         out = self.body(x)
#         out = F.relu(out)
#         return out




# class PAM_CAM_Layer(nn.Module):
#     def __init__(self, in_channels):
#         super(PAM_CAM_Layer, self).__init__()
#         self.pam = PAM_Module(in_channels)
#         self.cam = CAM_Module(in_channels)
#         self.basic1 = basicBlock_dual(in_channels=in_channels, out_channels=in_channels)   # conv11 + relu + conv33 + relu
#         self.basic2 = basicBlock_dual(in_channels=in_channels, out_channels=in_channels)
#         # 定义通道交互模块
#         self.channel_interaction = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
#             nn.BatchNorm2d(in_channels // 8),
#             nn.GELU(),
#             nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#         # 定义空间交互模块
#         self.spatial_interaction = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
#             nn.BatchNorm2d(in_channels // 16),
#             nn.GELU(),
#             nn.Conv2d(in_channels // 16, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#         # 深度卷积分支
#         self.dwconv = nn.Sequential(
#             nn.Conv2d(2 * in_channels , in_channels ,1),
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,groups=in_channels ),
#             nn.BatchNorm2d(in_channels),
#             nn.GELU()
#         )

#         self.atten_conv = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,1,1),
#                                         nn.BatchNorm2d(in_channels),
#                                         nn.ReLU())

#         self.final = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding = 1),
#                                    nn.BatchNorm2d(in_channels),
#                                    nn.ReLU())
#     def forward(self, x):  
#         short_cut = x
#         ### 注意力分支
#         # 通道注意力
#         x_cam = self.cam(x)
#         short_cut_cam = self.basic1(x_cam)
#         ci = self.channel_interaction(short_cut_cam)
        
#         # 空间注意力 
#         x_pam = self.pam(x)
#         short_cut_pam = self.basic2(x_pam)

#         si = self.spatial_interaction(short_cut_pam)

#         ### DWConv分支
#         feat = torch.cat([x_cam,x_pam],dim=1)
#         dw = self.dwconv(feat)
#         dw_ci = self.channel_interaction(dw)
#         dw_si = self.spatial_interaction(dw)

#         x_cam_m = ci * dw_si * short_cut_cam
#         x_pam_m = si * dw_ci * short_cut_pam

#         atten = self.atten_conv(x_cam_m + x_pam_m)
#         out = self.final(atten + short_cut)
#         return out





# # --------------------------------------------MSCF------------------------------
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class DWConv_P(nn.Module):
#     def __init__(self, in_channels, kernel_size):
#         super(DWConv_P, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, in_channels, 1, bias=False)
#         self.bn = nn.BatchNorm2d(in_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class PyramidPooling(nn.Module):
#     def __init__(self, in_channels):
#         super(PyramidPooling, self).__init__()
#         self.in_channels = in_channels
        
#         self.convs = nn.ModuleList([
#             DWConv_P(in_channels=in_channels, kernel_size=k) for k in (1, 3, 5, 7)
#         ])
        
#         self.attention = SELayer(in_channels * 4)
        
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(in_channels * 4, in_channels, 1, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True)
#         )
        
#     def forward(self, x):
#         short_cut = x
#         size = x.size()[2:]
#         feats = []

#         pooled_1x1 = F.avg_pool2d(x, kernel_size=1, stride=1, padding=0)
#         pooled_3x3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
#         pooled_5x5 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=0)
#         pooled_7x7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)
        

#         pools = [pooled_1x1, pooled_3x3, pooled_5x5, pooled_7x7]

#         for pool, conv in zip(pools, self.convs):
#             feat = conv(pool)
#             feat = F.interpolate(feat, size, mode='bilinear', align_corners=False)
#             feats.append(feat)
        
#         out = torch.cat(feats, dim=1)
#         out = self.attention(out)
#         out = self.bottleneck(out) + short_cut

        
#         return out




# # class PyramidPooling(nn.Module):
# #     def __init__(self, in_channels):
# #         super(PyramidPooling, self).__init__()
# #         # self.conv_1x1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, bias=False)
# #         # self.conv_3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, bias=False)
# #         # self.conv_5x5 = nn.Conv2d(in_channels, in_channels, 3, padding=5, dilation=5, bias=False)
# #         # self.conv_7x7 = nn.Conv2d(in_channels, in_channels, 3, padding=7, dilation=7, bias=False)
# #         self.conv_1x1 = nn.Conv2d(in_channels, in_channels, 1, padding=0, groups=in_channels, bias=False)
# #         self.conv_3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
# #         self.conv_5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False)
# #         self.conv_7x7 = nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels, bias=False)
        
# #         self.bn_1x1 = nn.BatchNorm2d(in_channels)
# #         self.bn_3x3 = nn.BatchNorm2d(in_channels)
# #         self.bn_5x5 = nn.BatchNorm2d(in_channels)
# #         self.bn_7x7 = nn.BatchNorm2d(in_channels)

# #         self.final_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
# #         self.final_bn = nn.BatchNorm2d(in_channels)
# #         self.final_activate = nn.ReLU(inplace=True)

# #     def forward(self, x):
# #         h, w = x.size(2), x.size(3)

# #         pooled_1x1 = F.avg_pool2d(x, kernel_size=1, stride=1, padding=0)
# #         pooled_3x3 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
# #         pooled_5x5 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=0)
# #         pooled_7x7 = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)

# #         out_1x1 = self.bn_1x1(self.conv_1x1(pooled_1x1))
# #         out_3x3 = F.interpolate(self.bn_3x3(self.conv_3x3(pooled_3x3)), size=(h, w), mode='bilinear', align_corners=True)
# #         out_5x5 = F.interpolate(self.bn_5x5(self.conv_5x5(pooled_5x5)), size=(h, w), mode='bilinear', align_corners=True)
# #         out_7x7 = F.interpolate(self.bn_7x7(self.conv_7x7(pooled_7x7)), size=(h, w), mode='bilinear', align_corners=True)

# #         return self.final_activate(self.final_bn(self.final_conv(out_1x1 + out_3x3 + out_5x5 + out_7x7)))



    


# class MSCFModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MSCFModule, self).__init__()
#         self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(in_channels // 4)
#         self.pyramid_pooling = PyramidPooling(in_channels // 4)
#         self.anisotropic_strip_pooling = PAM_CAM_Layer(in_channels // 4)
#         self.conv1x1_2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(in_channels)
#         self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         x = self.relu(self.bn1(self.conv1x1_1(x)))
#         pp_out = self.pyramid_pooling(x)
#         asp_out = self.anisotropic_strip_pooling(pp_out)
#         combined = asp_out
#         x = self.relu(residual + self.bn2(self.conv1x1_2(combined)))
#         x = self.relu(self.bn3(self.final_conv(x)))

#         return x



# class Block(nn.Module):

#     def __init__(self,
#                  dim,
#                  mlp_ratio=4.,
#                  drop=0.,
#                  drop_path=0.,
#                  act_layer=nn.GELU,
#                  norm_cfg=dict(type='BN')):
#         super().__init__()
#         self.norm1 = nn.BatchNorm2d(dim)
#         self.attn = MSCFModule(in_channels=dim , out_channels = dim)
#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = nn.BatchNorm2d(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim,
#                        hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer,
#                        drop=drop)
#         layer_scale_init_value = 1e-2
#         # self.layer_scale_1 = layer_scale_init_value * jt.ones((dim))
#         # self.layer_scale_2 = layer_scale_init_value * jt.ones((dim))
#         self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
#         self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.permute(0, 2, 1).view(B, C, H, W)
#         x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *self.mlp(self.norm2(x)))
#         x = x.view(B, C, N).permute(0, 2, 1)
#         return x




# class DWConv(nn.Module):

#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x):
#         x = self.dwconv(x)
#         return x



# class Feature_Incentive_Block(nn.Module):
#     def __init__(self, img_size=224, patch_size=3, stride=1, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
#         self.num_patches = self.H * self.W
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
#                               padding=(patch_size[0] // 2, patch_size[1] // 2))
#         self.norm = nn.LayerNorm(embed_dim)
#         self.act = nn.GELU()


#     def forward(self, x):
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.act(x)
#         x = self.norm(x)
#         return x, H, W



# class MSCAN(nn.Module):
#     def __init__(self,
#                  in_chans=3,
#                 #  embed_dims=[64, 128, 256, 512],
#                  embed_dims=[768],
#                 #  mlp_ratios=[4, 4, 4, 4],
#                  mlp_ratios=[4],
#                  drop_rate=0.,
#                  drop_path_rate=0.2,
#                 #  depths=[3, 4, 6, 3],
#                  depths=[1],
#                  num_stages=1,
#                  norm_cfg=dict(type='BN')):
#         super(MSCAN, self).__init__()

#         self.depths = depths
#         self.num_stages = num_stages

#         # dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0

#         # self.patch_embed = Feature_Incentive_Block(in_chans=in_chans,embed_dim=embed_dims[0])

#         for i in range(num_stages):
#             block = nn.ModuleList([
#                 Block(dim=in_chans,
#                     #   dim=embed_dims[i],
#                       mlp_ratio=mlp_ratios[i],
#                       drop=drop_rate,
#                       drop_path=dpr[cur + j],
#                       norm_cfg=norm_cfg) for j in range(depths[i])
#             ])
#             norm = nn.LayerNorm(in_chans)
#             cur += depths[i]

#             setattr(self, f"block{i + 1}", block)
#             setattr(self, f"norm{i + 1}", norm)

#         self.conv1x1 = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=1, stride=1)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)


#     def forward(self, x,H,W):
#         # B,C,H,W = x.shape
#         # x, H, W = self.patch_embed(x)

#         for i in range(self.num_stages):
#             block = getattr(self, f"block{i + 1}")
#             norm = getattr(self, f"norm{i + 1}")
#             for blk in block:
#                 x = blk(x, H, W)
#             # x = norm(x)
#             # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

#             # x = self.conv1x1(x)
#         return x



# ------------------------------------------------------------------------------------------------

__all__ = ['UDM']

import torch
import torch.nn as nn
import torch.nn.functional as F

# class UED(nn.Module):
#     def __init__(self, in_channels):
#         super(UED, self).__init__()
#         self.feq = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.feq(x)
        
#         # 防止 log(0) 错误，clamp 函数确保值不会小于 1e-6
#         x_top1 = x * torch.log(torch.clamp(x, min=1e-6))
#         x_bottom1 = torch.log(torch.tensor(x.shape[1], dtype=torch.float32))
        
#         # 计算不确定性
#         uncert = (-x_top1 / x_bottom1).sum(dim=1)
#         return uncert

# class UFFM(nn.Module):
#     def __init__(self, in_channels, kernel_size=3, epsilon=1e-6):
#         super(UFFM, self).__init__()
#         self.kernel_size = kernel_size
#         self.epsilon = epsilon

#         # 置信度统计层
#         self.fc_kernel = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
#         )
        
#         # 通道调制层
#         self.fc_channel_weight = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         # 深度卷积层
#         self.depthwise_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1, groups=in_channels, bias=False
#         )

#     def forward(self, fin, Um):
#         batch, channels, height, width = fin.shape
        
#         # 计算不确定性权重
#         Um_flat = Um.view(Um.size(0), Um.size(1), -1)
#         w = 1 - torch.exp(Um_flat) 
#         w = w / torch.sum(w, dim=-1, keepdim=True) 
#         w = w.transpose(1, 2)

#         # 计算全局统计量
#         fin_flat = fin.view(fin.size(0), fin.size(1), -1)
#         fcg = torch.bmm(fin_flat, w)
#         fcg = fcg.unsqueeze(-1)

#         # 使用 fcg 生成深度卷积核 K
#         kernel = self.fc_kernel(fcg)
        
#         # 卷积核归一化
#         kernel = kernel / (torch.sqrt(torch.sum(kernel ** 2, dim=(2, 3), keepdim=True) + self.epsilon))

#         # 使用归一化内核执行深度卷积
#         f1 = self.depthwise_conv(kernel)
#         f1 = f1 * fin

#         # 生成通道加权系数 v 并执行通道调制
#         channel_weight = self.fc_channel_weight(fcg)
#         f2 = fin * channel_weight

#         # 融合 f1 和 f2 得到最终的 fm
#         fm = f1 + f2
#         fout = fin + fm

#         return fout

# class UDM(nn.Module):
#     def __init__(self, in_channels):
#         super(UDM, self).__init__()
#         self.ueb = UED(in_channels)
#         self.feb = UFFM(in_channels)

#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # 计算不确定性图并确保其在合理范围
#         uncertainty_map = self.ueb(x).unsqueeze(1)
#         uncertainty_map = torch.clamp(uncertainty_map, min=0.0, max=1.0)  # 防止值超出范围
        
#         # 使用 UFFM 增强特征
#         enhanced_features = self.feb(x, uncertainty_map)
        
#         # 融合原始特征和增强特征
#         output = (1 - uncertainty_map) * x + uncertainty_map * enhanced_features

#         return output

class UED(nn.Module):
    def __init__(self, in_channels):
        super(UED, self).__init__()
        # 增加多层卷积，增强网络学习能力
        self.feq = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),  # 第一层卷积
            nn.LeakyReLU(0.2),  # 使用 LeakyReLU 激活避免梯度消失
            nn.Conv2d(in_channels // 2, in_channels, 3, 1, 1),  # 第二层卷积
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),  # 最后一层卷积输出不确定性图
            nn.Sigmoid()  # 输出 [0, 1] 之间的值
        )

    def forward(self, x):
        x = self.feq(x)

        # 计算不确定性图：H(x) = -Σ p(x) log(p(x))
        # 避免对 0 取对数，使用 clamp 来防止数值问题
        x_top1 = x * torch.log(torch.clamp(x, min=1e-6))
        uncertainty_map = -torch.sum(x_top1, dim=1)  # 对通道维度求和，得到不确定性图
        return uncertainty_map

class UFFM(nn.Module):
    def __init__(self, in_channels, kernel_size=3, epsilon=1e-6):
        super(UFFM, self).__init__()
        self.kernel_size = kernel_size
        self.epsilon = epsilon

        # 置信度统计层（生成卷积核）
        self.fc_kernel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )

        # 通道加权层
        self.fc_channel_weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()  # 用 Sigmoid 保证输出 [0, 1]
        )

        # 深度卷积层
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1, groups=in_channels, bias=False
        )

    def forward(self, fin, Um):
        batch, channels, height, width = fin.shape

        # 计算不确定性图的权重
        Um_flat = Um.view(Um.size(0), Um.size(1), -1)
        w = 1 - torch.exp(Um_flat)  # 对不确定性图取指数，生成权重
        w = w / torch.sum(w, dim=-1, keepdim=True)  # 对权重做归一化
        w = w.transpose(1, 2)

        # 计算全局统计量
        fin_flat = fin.view(fin.size(0), fin.size(1), -1)
        fcg = torch.bmm(fin_flat, w)  # 进行矩阵乘法，得到全局特征
        fcg = fcg.unsqueeze(-1)

        # 使用全连接生成卷积核
        kernel = self.fc_kernel(fcg)
        kernel = kernel / (torch.sqrt(torch.sum(kernel ** 2, dim=(2, 3), keepdim=True) + self.epsilon))  # 归一化卷积核

        # 使用深度卷积处理特征
        f1 = self.depthwise_conv(kernel)
        f1 = f1 * fin

        # 通道加权
        channel_weight = self.fc_channel_weight(fcg)
        f2 = fin * channel_weight

        # 融合 f1 和 f2
        fm = f1 + f2
        fout = fin + fm  # 输出最终的增强特征

        return fout



class UDM(nn.Module):
    def __init__(self, in_channels):
        super(UDM, self).__init__()
        self.ueb = UED(in_channels)  # 不确定性图生成模块
        self.feb = UFFM(in_channels)  # 特征增强模块

    def forward(self, x):
        B, C, H, W = x.shape

        # 计算不确定性图
        uncertainty_map = self.ueb(x).unsqueeze(1)
        uncertainty_map = torch.clamp(uncertainty_map, min=0.0, max=1.0)  # 保证不确定性图在合理范围内

        # 使用 UFFM 增强特征
        enhanced_features = self.feb(x, uncertainty_map)

        # 动态加权：融合原始特征与增强特征
        output = (1 - uncertainty_map) * x + uncertainty_map * enhanced_features

        return output
