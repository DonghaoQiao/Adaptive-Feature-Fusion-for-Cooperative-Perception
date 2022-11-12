
"""
Implementation of C-AdaFusion model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEFusion(nn.Module):
    def __init__(self, max_cav):
        super(SEFusion, self).__init__()
        self.max_cav = max_cav
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.ex = nn.Sequential(
            nn.Linear(self.max_cav*2, self.max_cav, bias=False),
            nn.Sigmoid(),
            nn.Linear(self.max_cav, self.max_cav, bias=False), 
            nn.ReLU(inplace=True)
        )
        self.conv3d = nn.Conv3d(self.max_cav, 1, 1, stride=1, padding=0)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        bs = len(record_len)
        split_x = self.regroup(x, record_len)
        out = []
        
        for xx in split_x:
            rl = xx.shape[0]
            xx = F.pad(xx, (0,0,0,0,0,0,0,self.max_cav-rl)).unsqueeze(0)
            out.append(xx)
        out = torch.cat(out, dim=0) # [bs, 256, 100, 352]
        x_avg = self.avgpool(out).view(bs, self.max_cav)
        x_max = self.maxpool(out).view(bs, self.max_cav)
        x_sq = torch.cat((x_avg, x_max), dim=1)
        x_ex = self.ex(x_sq).view(bs, self.max_cav, 1, 1, 1)
        out = out * x_ex.expand_as(out)
        out = F.relu(self.conv3d(out), inplace=True).squeeze(1)

        return out
