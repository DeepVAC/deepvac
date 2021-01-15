import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# introduced by enhanced CNN BEGIN
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, makeDivisible(channel // reduction, 8)),
            nn.ReLU(inplace=False),
            nn.Linear(makeDivisible(channel // reduction, 8), channel),
            hsigmoid(inplace=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=-1, groups=1):
        if padding == -1:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=False)
        )
# enhanced CNN END

# introduced by mobilenet series BEGIN
def makeDivisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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

class Conv2dBNHswish(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        if padding == -1:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHswish, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            hswish(inplace=False)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        hidden_dim = makeDivisible(inp * expand_ratio, 8)
        assert stride in [1, 2]
        assert kernel_size in [3,5]

        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dBNHswish(inp, hidden_dim, kernel_size=1))  # pw
        
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            hswish() if use_hs else nn.ReLU(inplace=False),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Identity(),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
# mobilenet series END





