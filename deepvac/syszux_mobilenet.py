import torch.nn as nn
import math
from .syszux_modules import makeDivisible,hswish,Conv2dBNHswish,InvertedResidual,initWeights

class MobileNetV3(nn.Module):
    def __init__(self, class_num=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        self.width_mult = width_mult
        self.class_num = class_num
        # setting of inverted residual blocks
        self.auditConfig()
        # building first layer
        input_channel = makeDivisible(16 * self.width_mult, 8)
        layers = [Conv2dBNHswish(3, input_channel, stride=2)]

        # building inverted residual blocks
        for k, t, c, use_se, use_hs, s in self.cfgs:
            exp_size = makeDivisible(input_channel * t, 8)
            output_channel = makeDivisible(c * self.width_mult, 8)
            layers.append(InvertedResidual(input_channel, output_channel, k, s, t, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = Conv2dBNHswish(input_channel, exp_size, kernel_size=1)
        self.fc_inp = exp_size
        self.initFc()
        initWeights(self)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def initFc(self):
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = makeDivisible(self.last_output_channel * self.width_mult, 8) if self.width_mult > 1.0 else self.last_output_channel
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_inp, output_channel),
            hswish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, self.class_num),
        )

    def auditConfig(self):
        self.last_output_channel = 1024
        self.cfgs = [
            # k, t, c, SE, HS, s 
            [3,    1,  16, True, False, 2],
            [3,  4.5,  24, False, False, 2],
            [3, 3.67,  24, False, False, 1],
            [5,    4,  40, True, True, 2],
            [5,    6,  40, True, True, 1],
            [5,    6,  40, True, True, 1],
            [5,    3,  48, True, True, 1],
            [5,    3,  48, True, True, 1],
            [5,    6,  96, True, True, 2],
            [5,    6,  96, True, True, 1],
            [5,    6,  96, True, True, 1],
        ]


class MobileNetV3Large(MobileNetV3):
    def auditConfig(self):
        self.last_output_channel = 1280
        self.cfgs = [
            # k, t, c, SE, HS, s 
            [3,   1,  16, False, False, 1],
            [3,   4,  24, False, False, 2],
            [3,   3,  24, False, False, 1],
            [5,   3,  40, True, False, 2],
            [5,   3,  40, True, False, 1],
            [5,   3,  40, True, False, 1],
            [3,   6,  80, False, True, 2],
            [3, 2.5,  80, False, True, 1],
            [3, 2.3,  80, False, True, 1],
            [3, 2.3,  80, False, True, 1],
            [3,   6, 112, True, True, 1],
            [3,   6, 112, True, True, 1],
            [5,   6, 160, True, True, 2],
            [5,   6, 160, True, True, 1],
            [5,   6, 160, True, True, 1]
        ]

class MobileNetV3Ocr(MobileNetV3):
    def auditConfig(self):
        self.last_output_channel = 1024
        self.cfgs = [
            # k, t, c, SE, HS, s 
            [3,    1,  16, True, False, 1],
            [3,  4.5,  24, False, False, (2,1)],
            [3, 3.67,  24, False, False, 1],
            [5,    4,  40, True, True, (2,1)],
            [5,    6,  40, True, True, 1],
            [5,    6,  40, True, True, 1],
            [5,    3,  48, True, True, 1],
            [5,    3,  48, True, True, 1],
            [5,    6,  96, True, True, (2,1)],
            [5,    6,  96, True, True, 1],
            [5,    6,  96, True, True, 1],
        ]
    def initFc(self):
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        x = self.features(x)
        x = self.conv(x)
        x = self.pool(x)

        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"

        x = x.squeeze(2)        # b *c * width
        x = x.permute(2, 0, 1)    # [w, b, c]
        return x
