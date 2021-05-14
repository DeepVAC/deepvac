from collections import OrderedDict
import torch.nn as nn
from .activate_layer import hswish

class Conv2dBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes)
        )

class Conv2dBNWithName(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNWithName, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))
        
class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=True)
        )

class Conv2dBNPReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNPReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.PReLU(out_planes)
        )

class Conv2dBnAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False, act=True):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBnAct, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.SiLU() if (act is True) else (act if isinstance(act, nn.Module) else nn.Identity())
        )

class Conv2dBNHardswish(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHardswish, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.Hardswish()
        )

class Conv2dBNHswish(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHswish, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            hswish(inplace=True)
        )

class DepthWiseConv2d(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, kernel_size: int, stride: int, padding: int, groups: int, residual: bool=False, bias: bool=False):
        super(DepthWiseConv2d, self).__init__()
        self.conv1 = Conv2dBNPReLU(inplanes, groups, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2dBNPReLU(groups, groups, kernel_size, stride, padding, groups)
        self.conv3 = nn.Conv2d(groups, outplanes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes)
        self.residual = residual

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        if self.residual:
            x = identity + x
        return x

class Conv2dBNLeakyReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1, leaky=0, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        #leaky = 0.1 if out_planes <= 64 else 0
        super(Conv2dBNLeakyReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(negative_slope=leaky, inplace=True)
        )
