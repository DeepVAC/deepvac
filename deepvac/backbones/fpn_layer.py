import torch.nn as nn
import torch.nn.functional as F
from .conv_layer import Conv2dBNLeakyReLU

class FPN(nn.Module):
    def __init__(self, in_planes: list, out_planes: int):
        super(FPN,self).__init__()

        leaky = 0.1 if out_planes <= 64 else 0
        self.conv1 = Conv2dBNLeakyReLU(in_planes[0], out_planes, kernel_size=1, padding=0, leaky=leaky)
        self.conv2 = Conv2dBNLeakyReLU(in_planes[1], out_planes, kernel_size=1, padding=0, leaky=leaky)
        self.conv3 = Conv2dBNLeakyReLU(in_planes[2], out_planes, kernel_size=1, padding=0, leaky=leaky)

        self.conv4 = Conv2dBNLeakyReLU(out_planes, out_planes, padding=1, leaky=leaky)
        self.conv5 = Conv2dBNLeakyReLU(out_planes, out_planes, padding=1, leaky=leaky)

    def forward(self, input):
        output1 = self.conv1(input[0])
        output2 = self.conv2(input[1])
        output3 = self.conv3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.conv5(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.conv4(output1)
        out = [output1, output2, output3]
        return out