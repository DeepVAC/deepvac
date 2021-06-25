import torch
import torch.nn as nn
from .conv_layer import *


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes: int, outplanes: int, stride: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2dBNReLU(in_planes=inplanes, out_planes=outplanes, kernel_size=1)
        self.conv2 = Conv2dBNReLU(in_planes=outplanes, out_planes=outplanes, kernel_size=3, stride=stride)

        outplanes_after_expansion = outplanes * self.expansion
        self.conv3 = nn.Conv2d(outplanes, outplanes_after_expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outplanes_after_expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = None

        if stride != 1 or inplanes != outplanes_after_expansion:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, outplanes_after_expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outplanes_after_expansion))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class BottleneckIR(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, stride: int):
        super(BottleneckIR, self).__init__()

        self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride) if inplanes == outplanes else nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride ,bias=False), nn.BatchNorm2d(outplanes))

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1 ,bias=False),
            nn.PReLU(outplanes),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1 ,bias=False),
            nn.BatchNorm2d(outplanes)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

