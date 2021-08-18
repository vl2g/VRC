import torch.nn as nn
import torch


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sub_sp, sub_cls, sub_fc, ob_sp, ob_cls, ob_fc):
        sub_emb = torch.cat([sub_sp, sub_cls, sub_fc], dim=1)
        ob_emb = torch.cat([ob_sp, ob_cls, ob_fc], dim=1)

        vr_emb = torch.cat([sub_emb, ob_emb], dim=1)
        return vr_emb
