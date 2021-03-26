import numpy as np
import torch.nn as nn
import torch

#function begin
def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(anynet_slope, anynet_initial_width, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert anynet_slope >= 0 and anynet_initial_width > 0 and w_m > 1 and anynet_initial_width % q == 0
    ws_cont = np.arange(d) * anynet_slope + anynet_initial_width
    ks = np.round(np.log(ws_cont / anynet_initial_width) / np.log(w_m))
    ws = anynet_initial_width * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont
#function end

class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, class_num):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, class_num, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b, eps=1e-5, momentum=0.1)
        self.b_relu = nn.ReLU(inplace=True)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
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

    def __init__(self, w_in, w_out):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class RegNet(nn.Module):
    def __init__(self, anynet_slope, anynet_initial_width, w_m, d, group_w, class_num=0,stem_w = 32,se_r = 0.25):
        super(RegNet, self).__init__()
        ws, num_stages, _, _ = generate_regnet(anynet_slope, anynet_initial_width, w_m, d)
        # Convert to per stage format
        width_per_stage, depth_per_stage = get_stages_from_blocks(ws, ws)
        # Use the same gw, bm and ss for each stage
        # GROUP_W: 8 for 200MF
        group_w_per_stage = [group_w for _ in range(num_stages)]
        bot_muls_per_stage = [1.0 for _ in range(num_stages)]
        stride_per_stage = self.auditConfig(num_stages)
        assert(len(stride_per_stage) == num_stages, "configure stride_per_stage error!")
        # Adjust the compatibility of ws and gws
        width_per_stage, group_w_per_stage = adjust_ws_gs_comp(width_per_stage, bot_muls_per_stage, group_w_per_stage)
        # Generate dummy bot muls and gs for models that do not use them
        bot_muls_per_stage = bot_muls_per_stage if bot_muls_per_stage else [None for _d in depth_per_stage]
        group_w_per_stage = group_w_per_stage if group_w_per_stage else [None for _d in depth_per_stage]
        stage_params = list(zip(depth_per_stage, width_per_stage, stride_per_stage, bot_muls_per_stage, group_w_per_stage))

        self.stem = SimpleStemIN(3, stem_w)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, d, ResBottleneckBlock, bm, gw, se_r))
            prev_w = w

        if class_num > 0:
            self.head = AnyHead(w_in=prev_w, class_num=class_num)
    
    def auditConfig(self, num_stages):
        stride_per_stage = [2 for _ in range(num_stages)]
        return stride_per_stage

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class RegNetSmall(RegNet):
    def __init__(self, class_num):
        super(RegNetSmall, self).__init__(27.89, 48, 2.09, 16, 8, class_num)

class RegNetMedium(RegNet):
    def __init__(self, class_num):
        super(RegNetMedium, self).__init__(20.71, 48, 2.65, 27, 24, class_num)

class RegNetLarge(RegNet):
    def __init__(self, class_num):
        super(RegNetLarge, self).__init__(31.41, 96, 2.24, 22, 64, class_num)

class RegNetOCR(RegNet):
    def __init__(self):
        super(RegNetOCR, self).__init__(27.89, 48, 2.09, 16, 8, 0)

    def auditConfig(self, num_stages):
        stride_per_stage = [(2,1),(2,1),(2,1),2]
        return stride_per_stage

if __name__ == '__main__':
    model = RegNetSmall(1000)
    print(model)
    x = torch.randn((1,3,32,320))
    print(model(x).shape)
