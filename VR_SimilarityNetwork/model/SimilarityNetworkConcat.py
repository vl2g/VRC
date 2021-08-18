# Defining Network
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityNetworkConcat(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        input_size = model_config.SimilarityNetworkConcatInputSize * 2
        self.fc1 = nn.Linear(input_size, 7306)
        self.fc2 = nn.Linear(input_size, 7306)    
        self.fc3 = nn.Linear(7306, 1)

    
    def forward(self,rela1, rela2):
        # Assuming that both relations are of shape [batch_size, 500], we keep the axis as 1 if shape is not like this please change axis.
        x = torch.cat((rela1, rela2))
        x1 = F.tanh((self.fc1(x)))
        x2 = torch.sigmoid((self.fc2(x)))
        x = (x1 * x2 ) 
        x = x + ((rela1 + rela2)/2)
        x = self.fc3(x)
        return x

