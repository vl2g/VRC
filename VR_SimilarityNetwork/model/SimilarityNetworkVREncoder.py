# Defining Network
from utils.utils import load_config_file
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
print(use_cuda)
cuda = torch.device('cuda')


MODEL_CONFIG_PATH = "/DATA/trevant/Vaibhav/tempVRC/VR_SimilarityNetwork/configs/model_config.yaml"

model_config = load_config_file(MODEL_CONFIG_PATH)


class SimilarityNetworkVREncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        input_size = model_config.SimilarityNetworkVREncoderInputSize
        self.fc1 = nn.Linear(input_size, 500)
        self.fc2 = nn.Linear(input_size, 500)    
        self.fc3 = nn.Linear(500, 1)
        self.tan_bn = nn.BatchNorm1d(500)
        self.sig_bn = nn.BatchNorm1d(500)
    
    def forward(self,rela1, rela2):
        # Assuming that both relations are of shape [batch_size, 500], we keep the axis as 1 if shape is not like this please change axis.
        x = torch.cat((rela1, rela2))
        x1 = F.tanh((self.fc1(x)))
        x2 = torch.sigmoid((self.fc2(x)))
        x = (x1 * x2 ) + ((rela1 + rela2)/2)
        x = self.fc3(x)
        return x
