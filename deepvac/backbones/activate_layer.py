import torch.nn as nn

class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = x * self.relu(x+3) / 6
        return out

class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = self.relu(x+3) / 6
        return out