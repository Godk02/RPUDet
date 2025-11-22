import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import math

__all__ = ['RelationModule']


# [K,4] -> [K,K,4] # get the pairwise box geometric feature
def geometric_encoding_single(boxes):
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
    w = x2 - x1
    h = y2 - y1
    center_x = 0.5 * (x1+x2)
    center_y = 0.5 * (y1+y2)

    # [K,K]
    delta_x = center_x - torch.transpose(center_x,0,1)
    delta_x = delta_x / w
    delta_x = torch.log(torch.abs(delta_x).clamp(1e-3))

    delta_y = center_y - torch.transpose(center_y,0,1)
    delta_y = delta_y / w
    delta_y = torch.log(torch.abs(delta_y).clamp(1e-3))

    delta_w = torch.log(w / torch.transpose(w,0,1))

    delta_h = torch.log(h / torch.transpose(h,0,1))

    # [K,K,4]
    output = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=2)

    return output

def geometric_encoding_batch(boxes, bs):
    '''
    boxes: tensor[bs*roi_num,4]
    bs: batch size
    '''

    boxes = boxes.reshape(bs,-1,4)
    bbox_encoding_batch = []
    for bs_id in range(0,bs):
        bbox_encoding = geometric_encoding_single(boxes[bs_id].squeeze())
        bbox_encoding_batch.append(bbox_encoding)
    # [bs,roi_num,roi_num,4]
    output = torch.stack(bbox_encoding_batch, dim=0)

    return output



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=groups, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        # 定义默认激活函数
        self.default_act = nn.SiLU()  # 你可以根据需要选择其他激活函数

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class RelationModule(nn.Module):
    """
    Simple layer for self-attention, 

    Input:
    group: the head number of relation module
    geo_feat_dim: embedding the position

    Output:
    [bs*roi_num,1024]
    """

    def __init__(self, box_feat_dim=1024, group=16, geo_feat_dim=64):
        super(RelationModule, self).__init__()
        self.box_feat_dim = box_feat_dim
        self.group = group
        self.geo_feat_dim = geo_feat_dim
        self.group_feat_dim = int(box_feat_dim / group) # 1024/16=64

        self.tanh = nn.Tanh()
        self.geo_emb_fc = nn.Linear(4, geo_feat_dim) # 4->64
        self.box_geo_conv = ConvModule(geo_feat_dim, group, 1) # 1x1 conv
        self.query_fc = nn.Linear(box_feat_dim, box_feat_dim) 
        self.key_fc = nn.Linear(box_feat_dim, box_feat_dim)
        self.group_conv = ConvModule(group*box_feat_dim, box_feat_dim, 1, groups=group) # 16*1024->1024


    def forward(self, box_appearance_feat, boxes, bs):
        roi_num = int(box_appearance_feat.size(0)/bs)

        # [K,4] -> [bs,roi_num,roi_num,4]
        # given the absolute box, get the pairwise relative geometric coordinates
        box_geo_encoded = geometric_encoding_batch(boxes,bs)
        # [bs,roi_num,roi_num,4] -> [bs,roi_num,roi_num,64]
        box_geo_feat = self.tanh(self.geo_emb_fc(box_geo_encoded))
        
        # [bs,64,roi_num,roi_num]
        box_geo_feat = box_geo_feat.permute(0,3,1,2)  

        # [bs,16,roi_num,roi_num]
        box_geo_feat_wg = self.box_geo_conv(box_geo_feat)
        # [bs,roi_num,16,roi_num]
        box_geo_feat_wg = box_geo_feat_wg.permute(0,2,1,3)

        # now we get the appearance stuff
        # [bs,roi_num,1024]
        box_appearance_feat = box_appearance_feat.reshape(bs,-1,self.box_feat_dim)
        query = self.query_fc(box_appearance_feat)
        # split head
        # [bs,roi_num,16,1024/16]
        query = query.reshape(bs, -1, self.group, self.group_feat_dim)
        query = query.permute(0, 2, 1, 3) # [bs,16,roi_num,1024/16]

        key = self.key_fc(box_appearance_feat)
        # split head
        # [bs,roi_num,16,1024/16]
        key = key.reshape(bs, -1, self.group, self.group_feat_dim)
        key = key.permute(0, 2, 1, 3)  # [bs,16,roi_num,1024/16]

        value = box_appearance_feat

        key = key.permute(0, 1, 3, 2)  # [bs,16,1024/16,roi_num]
        # [bs,16,roi_num,1024/16]*[bs,16,1024/16,roi_num] ->[bs,16,roi_num,roi_num]
        logits = torch.matmul(query, key)
        logits_scaled = (1.0 / math.sqrt(self.group_feat_dim)) * logits
        logits_scaled = logits_scaled.permute(0, 2, 1, 3)  # [bs,roi_num,16,roi_num]

        # [bs,roi_num,16,roi_num]
        weighted_logits = torch.log(box_geo_feat_wg.clamp(1e-6)) + logits_scaled
        weighted_softmax = F.softmax(weighted_logits,dim=3)

        # need to reshape for matmul [bs,roi_num*16,roi_num]
        weighted_softmax = weighted_softmax.reshape(bs, roi_num*self.group, roi_num)

        # [bs,roi_num*16,roi_num] * [bs,roi_num,1024] -> [bs,roi_num*16,1024]
        output = torch.matmul(weighted_softmax, value)

        # [bs,roi_num,16*1024,1,1]
        output = output.reshape(bs, -1, self.group*self.box_feat_dim,1,1)

        out_batch = []
        for bs_id in range(0,bs):
            out_single = self.group_conv(output[bs_id].squeeze(0))
            out_batch.append(out_single)

        output = torch.stack(out_batch, dim=0)
        output = output.reshape(bs*roi_num,self.box_feat_dim)

        return output