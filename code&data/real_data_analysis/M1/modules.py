import torch.nn as nn # type: ignore
from torch.utils.data import Dataset # type: ignore
import torch.nn.functional as F # type: ignore


class Net_s(nn.Module):

    def __init__(self, indim, h, outdim):
        super(Net_s, self).__init__()
        self.h1 = nn.Linear(indim, 2*h)
        self.h2 = nn.Linear(2*h, h)  
        self.out = nn.Linear(h, outdim)
       

    def forward(self, input):
        c1 = F.relu(self.h1(input))
        c2 = F.relu(self.h2(c1))
        output = self.out(c2)
        return output
    
class data_set(Dataset):

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, id):

        feature = self.feature[id]
        label = self.label[id]
    
        return feature, label