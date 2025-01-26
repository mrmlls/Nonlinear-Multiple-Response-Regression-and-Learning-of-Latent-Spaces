from torch.utils.data import Dataset # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch





class Net_s(nn.Module):

    def __init__(self, k, q, p, h):
        super(Net_s, self).__init__()
        self.b = nn.Linear(p, k, bias=False)  
        self.h1 = nn.Linear(k, h)
        self.out = nn.Linear(h, q)

    def forward(self, input):
        c1 = F.relu(self.b(input))
        c2 = F.relu(self.h1(c1))
        output = self.out(c2)
        return output

    
class dataset(Dataset):

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, id):

        feature = self.feature[id]
        label = self.label[id]
        return feature, label