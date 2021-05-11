import torch
import torch.nn as nn
from .conv_layer import Conv2dBN, Conv2dBNLeakyReLU

class SSH(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(SSH, self).__init__()
        assert  out_planes % 4 == 0
        leaky = 0.1 if out_planes <= 64 else 0
        self.conv3X3 = Conv2dBN(in_planes, out_planes//2, padding=1)

        self.conv5X5_1 = Conv2dBNLeakyReLU(in_planes, out_planes//4, padding=1, leaky=leaky)
        self.conv5X5_2 = Conv2dBN(out_planes//4, out_planes//4, padding=1)

        self.conv7X7_2 = Conv2dBNLeakyReLU(out_planes//4, out_planes//4, padding=1, leaky=leaky)
        self.conv7x7_3 = Conv2dBN(out_planes//4, out_planes//4, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = self.relu(out)
        return out