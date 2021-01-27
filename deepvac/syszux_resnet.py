import torch
import torch.nn as nn
from .syszux_modules import initWeightsKaiming, BasicBlock, Bottleneck, Conv2dBNReLU

class ResNet18(nn.Module):
    def __init__(self, class_num: int = 1000):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.class_num = class_num
        self.auditConfig()
        self.conv1 = Conv2dBNReLU(in_planes=3, out_planes=self.inplanes, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        #init the 4 layers
        for outp, layer_num, stride in self.cfgs:
            layers.append(self.block(self.inplanes, outp, stride))
            self.inplanes = outp * self.block.expansion
            for _ in range(1, layer_num):
                layers.append(self.block(self.inplanes, outp))

        self.layer = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.initFc()
        initWeightsKaiming(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer(x)
        x = self.avgpool(x)
        return self.forward_cls(x)

    def forward_cls(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def initFc(self):
        self.fc = nn.Linear(512 * self.block.expansion, self.class_num)

    def auditConfig(self):
        self.block = BasicBlock
        self.cfgs = [
            # outp, layer_num, s
            [64,   2,  1],
            [128,  2,  2],
            [256,  2,  2],
            [512,  2,  2]
        ]

class ResNet34(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet34, self).__init__(class_num)

    def auditConfig(self):
        self.block = BasicBlock
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  4,  2],
            [256,  6,  2],
            [512,  3,  2]
        ]
    
class ResNet50(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet50, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  4,  2],
            [256,  6,  2],
            [512,  3,  2]
        ]

class ResNet101(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet101, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  4,  2],
            [256,  23,  2],
            [512,  3,  2]
        ]

class ResNet152(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet152, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  8,  2],
            [256,  36,  2],
            [512,  3,  2]
        ]

class ResNet18OCR(ResNet18):
    def __init__(self):
        super(ResNet18OCR, self).__init__()

    def auditConfig(self):
        self.block = BasicBlock
        self.cfgs = [
            [64,   2,  1],
            [128,  2,  (2,1)],
            [256,  2,  (2,1)],
            [512,  2,  (2,1)]
        ]

    def initFc(self):
        self.avgpool = nn.AvgPool2d((2,2))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        x = self.avgpool(x)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2) # b *512 * width
        x = x.permute(2, 0, 1)  # [w, b, c]
        return x
