import torch
import torch.nn as nn
from .conv_layer import Conv2dBNHardswish

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_planes, out_planes, pool_kernel_size=(5, 9, 13)):
        super(SPP, self).__init__()
        hidden_planes = in_planes // 2  # hidden channels
        self.conv1 = Conv2dBNHardswish(in_planes, hidden_planes, 1, 1)
        self.conv2 = Conv2dBNHardswish(hidden_planes * (len(pool_kernel_size) + 1), out_planes, 1, 1)
        self.pool_list = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_kernel_size])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pool_list], 1))



