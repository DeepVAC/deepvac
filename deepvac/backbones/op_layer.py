from typing import List
import torch
import torch.nn as nn

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, eps=1e-5, momentum=0.1, padding=None, groups=1):
        super(Focus, self).__init__()
        self.conv = Conv2dBNHardswish(in_planes * 4, out_planes, kernel_size, stride, eps, momentum, padding, groups)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))