# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of F-cooper maxout fusing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFusion(nn.Module):
    def __init__(self, max_cav):
        super(DFusion, self).__init__()
        self.max_cav = max_cav
        # self.conv3d = nn.Conv3d(self.max_cav, 1, 3, stride=1, padding=1)
        self.conv3d = nn.Sequential(
            nn.Conv3d(self.max_cav, 1, 3, stride=1, padding=1), 
            # nn.BatchNorm3d(1, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.Conv3d(8, 1, 3, stride=1, padding=1), 
            # nn.BatchNorm3d(1, eps=1e-3, momentum=0.01),
            # nn.ReLU(inplace=True),
        )
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        # x: [8, 256, 100, 352]
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []
        for xx in split_x:
            rl = xx.shape[0]
            # if rl < self.max_cav:
            #     tmp_x = torch.zeros((self.max_cav-rl, *xx.shape[1:])).cuda()
            #     xx = torch.cat([xx, tmp_x])
            xx = F.pad(xx, (0,0,0,0,0,0,0,self.max_cav-rl)).unsqueeze(0)
            out.append(xx)
        out = torch.cat(out, dim=0) # [bs, 256, 100, 352]
        out = self.conv3d(out).squeeze(1)
        return out
