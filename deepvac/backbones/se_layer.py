import torch.nn as nn
from .module_helper import makeDivisible
from .activate_layer import hsigmoid

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, makeDivisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(makeDivisible(channel // reduction, 8), channel),
            hsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y