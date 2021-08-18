import torch.nn as nn
import torch


class VTransE(nn.Module):
    def __init__(self, index_sp=True, index_cls=True, num_pred=100, output_size=500, input_size=500):
        super().__init__()
        self.index_sp = index_sp
        self.index_cls = index_cls
        if(index_sp == True):
            input_size += 4  # 2028
        if(index_cls == True):
            input_size += 1601  # 1601
        self.roi_contract_sub = nn.Linear(2048, 500)
        self.roi_contract_obj = nn.Linear(2048, 500)

        self.relu = nn.ReLU()
        self.sub_fc_layer = nn.Linear(input_size, output_size)
        self.obj_fc_layer = nn.Linear(input_size, output_size)
        self.rela_layer = nn.Linear(output_size, num_pred)

    def forward(self, sub_sp, sub_cls, sub_fc, ob_sp, ob_cls, ob_fc):
        sub_fc = self.relu(self.roi_contract_sub(sub_fc))
        ob_fc = self.relu(self.roi_contract_obj(ob_fc))
        if self.index_sp:
            sub_fc = torch.cat([sub_fc, sub_sp], axis=1)
            ob_fc = torch.cat([ob_fc, ob_sp], axis=1)
        if self.index_cls:
            sub_fc = torch.cat([sub_fc, sub_cls], axis=1)
            ob_fc = torch.cat([ob_fc, ob_cls], axis=1)
        sub_emb = torch.relu(self.sub_fc_layer(sub_fc))
        ob_emb = torch.relu(self.obj_fc_layer(ob_fc))

        vr_emb = ob_emb - sub_emb

        rela_score = self.rela_layer(vr_emb)

        return rela_score, vr_emb

    def forward_inference(self, sub_sp, sub_cls, sub_fc, ob_sp, ob_cls, ob_fc):
        sub_fc = self.relu(self.roi_contract_sub(sub_fc))
        ob_fc = self.relu(self.roi_contract_obj(ob_fc))
        if self.index_sp:
            sub_fc = torch.cat([sub_fc, sub_sp])
            ob_fc = torch.cat([ob_fc, ob_sp])
        if self.index_cls:
            sub_fc = torch.cat([sub_fc, sub_cls])
            ob_fc = torch.cat([ob_fc, ob_cls])
        sub_emb = torch.relu(self.sub_fc_layer(sub_fc))
        ob_emb = torch.relu(self.obj_fc_layer(ob_fc))

        vr_emb = ob_emb - sub_emb

        rela_score = self.rela_layer(vr_emb)

        return rela_score, vr_emb