import numpy as np
import torch
import torch.nn as nn
from .weights_init import initWeightsKaiming
#function begin

def adjust_block_compatibility(width_per_stage, bot_muls_per_stage, group_w_per_stage):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(width_per_stage) == len(bot_muls_per_stage) == len(group_w_per_stage)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(width_per_stage, bot_muls_per_stage, group_w_per_stage))
    assert all(b < 1 or b % 1 == 0 for b in bot_muls_per_stage)
    vs = [int(max(1, w * b)) for w, b in zip(width_per_stage, bot_muls_per_stage)]
    group_w_per_stage = [int(min(g, v)) for g, v in zip(group_w_per_stage, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(group_w_per_stage, bot_muls_per_stage)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    width_per_stage = [int(v / b) for v, b in zip(vs, bot_muls_per_stage)]
    assert all(w * b % g == 0 for w, b, g in zip(width_per_stage, bot_muls_per_stage, group_w_per_stage))
    return width_per_stage, bot_muls_per_stage, group_w_per_stage

def generate_regnet(anynet_slope, anynet_initial_width, channel_controller, depth, divisor=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert anynet_slope >= 0 and anynet_initial_width > 0 and channel_controller > 1 and anynet_initial_width % divisor == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(depth) * anynet_slope + anynet_initial_width
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / anynet_initial_width) / np.log(channel_controller))
    ws_all = anynet_initial_width * np.power(channel_controller, ks)
    ws_all = np.round(np.divide(ws_all, divisor)).astype(int) * divisor
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont

class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, in_planes, class_num, head_w):
        super(AnyHead, self).__init__()
        self.head_w = head_w
        if head_w>0:
            self.conv = nn.Conv2d(in_planes, head_w, 1, stride=1, padding=0, groups=1, bias=False)
            self.bn = nn.BatchNorm2d(head_w, eps=1e-5, momentum=0.1)
            self.relu = nn.ReLU(inplace=True)
            in_planes = head_w
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, class_num, bias=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x))) if self.head_w > 0 else x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, in_planes, se_planes):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(in_planes, se_planes, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_planes, in_planes, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, in_planes, out_planes, stride, bottom_mul, group_width, se_ratio):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(out_planes * bottom_mul))
        g = w_b // group_width
        self.a = nn.Conv2d(in_planes, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        if se_ratio:
            se_planes = int(round(in_planes * se_ratio))
            self.se = SE(w_b, se_planes)
        self.c = nn.Conv2d(w_b, out_planes, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, in_planes, out_planes, stride, bottom_mul=1.0, group_width=1, se_ratio=None):
        super(ResBottleneckBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (in_planes != out_planes) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(in_planes, out_planes, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1)
        self.f = BottleneckTransform(in_planes, out_planes, stride, bottom_mul, group_width, se_ratio)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, in_planes, out_planes):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, in_planes, out_planes, stride, depth, block_fun, bottom_mul, group_width, se_ratio):
        super(AnyStage, self).__init__()
        for i in range(depth):
            b_stride = stride if i == 0 else 1
            b_w_in = in_planes if i == 0 else out_planes
            name = "b{}".format(i + 1)
            self.add_module(name, block_fun(b_w_in, out_planes, b_stride, bottom_mul, group_width, se_ratio))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class RegNet(nn.Module):
    def __init__(self, anynet_slope, anynet_initial_width, channel_controller, depth, group_w, class_num=0, head_w=0):
        super(RegNet, self).__init__()
        stem_w = 32
        se_ratio = 0.25
        width_per_stage, depth_per_stage = generate_regnet(anynet_slope, anynet_initial_width, channel_controller, depth)[0:2]
        group_w_per_stage = [group_w for _ in width_per_stage]
        bot_muls_per_stage = [1.0 for _ in width_per_stage]
        stride_per_stage = self.auditConfig(width_per_stage)
        width_per_stage, bot_muls_per_stage, group_w_per_stage = adjust_block_compatibility(width_per_stage, bot_muls_per_stage, group_w_per_stage)
        stage_params = list(zip(depth_per_stage, width_per_stage, stride_per_stage, bot_muls_per_stage, group_w_per_stage))

        self.stem = SimpleStemIN(3, stem_w)
        prev_w = stem_w
        for i, (depth, w, s, bottom_mul, group_width) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, depth, ResBottleneckBlock, bottom_mul, group_width, se_ratio))
            prev_w = w

        if class_num > 0:
            self.head = AnyHead(in_planes=prev_w, class_num=class_num, head_w=head_w)

        initWeightsKaiming(self)
    
    def auditConfig(self, width_per_stage):
        stride_per_stage = [2 for _ in width_per_stage]
        return stride_per_stage

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class RegNetSmall(RegNet):
    def __init__(self, class_num=0, head_w=0):
        super(RegNetSmall, self).__init__(27.89, 48, 2.09, 16, 8, class_num, head_w)

class RegNetMedium(RegNet):
    def __init__(self, class_num=0, head_w=0):
        super(RegNetMedium, self).__init__(20.71, 48, 2.65, 27, 24, class_num, head_w)

class RegNetLarge(RegNet):
    def __init__(self, class_num=0, head_w=0):
        super(RegNetLarge, self).__init__(31.41, 96, 2.24, 22, 64, class_num, head_w)

class RegNetOCR(RegNet):
    def __init__(self, class_num=0, head_w=0):
        super(RegNetOCR, self).__init__(27.89, 48, 2.09, 16, 8, class_num, head_w)

    def auditConfig(self, width_per_stage):
        stride_per_stage = [(2,1),(2,1),(2,1),2]
        return stride_per_stage

if __name__ == '__main__':
    model = RegNetOCR()
    LOG.logI(model)
    x = torch.randn((1,3,32,320))
    LOG.logI(model(x).shape)
