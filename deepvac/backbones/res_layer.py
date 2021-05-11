import torch.nn as nn
from .module_helper import makeDivisible
from .conv_layer import Conv2dBNReLU, Conv2dBNHswish
from .se_layer import SELayer
from .activate_layer import hswish

class ResBnBlock(nn.Module):
     def __init__(self, in_channel, expand_channel, out_channel, kernel_size, stride, shortcut=True):
         super(ResBnBlock, self).__init__()
         self.layer = nn.Sequential(
                 Conv2dBNReLU(in_channel, expand_channel, 1, 1),
                 Conv2dBNReLU(expand_channel, expand_channel, kernel_size, stride, kernel_size//2, expand_channel),
                 Conv2dBN(expand_channel, out_channel, 1, 1),
                 )
         self.shortcut = shortcut and (in_channel == out_channel)

     def forward(self, x):
         return self.layer(x) + x if self.shortcut else self.layer(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, use_se, use_hs, padding=None):
        super(InvertedResidual, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        hidden_dim = makeDivisible(inp * expand_ratio, 8)
        assert stride in [1, 2, (2, 1)]
        assert kernel_size in [3,5]

        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dBNHswish(inp, hidden_dim, kernel_size=1))  # pw

        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            hswish() if use_hs else nn.ReLU(inplace=True),
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

class InvertedResidualFacialKpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion_factor: int=6, kernel_size: int=3, stride: int=2, padding: int=1,
                 is_residual: bool=True):
        super(InvertedResidualFacialKpBlock, self).__init__()
        assert stride in [1, 2]

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, padding, 1,
                      groups=in_channels * expansion_factor, bias=False),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * expansion_factor, out_channels, 1,
                      bias=False),
            nn.BatchNorm2d(out_channels))

        self.is_residual = is_residual if stride == 1 else False
        self.is_conv_res = False if in_channels == out_channels else True
        #todo
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block