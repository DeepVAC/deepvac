import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# introduced by enhanced CNN BEGIN
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, makeDivisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(makeDivisible(channel // reduction, 8), channel),
            hsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=True)
        )

class Conv2dBNPReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNPReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.PReLU(out_planes)
        )

def initWeights(civilnet):
    for m in civilnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

class Conv2dBNHardswish(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHardswish, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.Hardswish()
        )

class BottleneckStd(nn.Module):
    # Standard bottleneck
    def __init__(self, in_planes, out_planes, groups=1, shortcut=True, expansion=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        hidden_planes = int(out_planes * expansion)  # hidden channels
        self.conv1 = Conv2dBNHardswish(in_planes, hidden_planes, 1, 1)
        self.conv2 = Conv2dBNHardswish(hidden_planes, out_planes, 3, 1, groups=groups)
        self.add = shortcut and in_planes == out_planes

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self, in_planes, out_planes, bottle_std_num=1, shortcut=True, groups=1, expansion=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        hidden_planes = int(out_planes * expansion)  # hidden channels
        self.conv1 = Conv2dBNHardswish(in_planes, hidden_planes, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, hidden_planes, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_planes, hidden_planes, 1, 1, bias=False)
        self.conv4 = Conv2dBNHardswish(2 * hidden_planes, out_planes, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hidden_planes)  # applied to cat(conv2, conv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.std_bottleneck_list = nn.Sequential(*[BottleneckStd(hidden_planes, hidden_planes, groups=groups, shortcut=shortcut, expansion=1.0) for _ in range(bottle_std_num)])

    def forward(self, x):
        y1 = self.conv3(self.std_bottleneck_list(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

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

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=None, groups=1):
        super(Focus, self).__init__()
        self.conv = Conv2dBNHardswish(in_planes * 4, out_planes, kernel_size, stride, padding, groups)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
# enhanced CNN END

# introduced by mobilenet series BEGIN
def makeDivisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = x * self.relu(x+3) / 6
        return out

class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        out = self.relu(x+3) / 6
        return out

class Conv2dBNHswish(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, groups=1):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(Conv2dBNHswish, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            hswish(inplace=True)
        )

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
# mobilenet series END

# introduced by resnet begin
class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, outplanes: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dBNReLU(in_planes=inplanes, out_planes=outplanes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = None
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outplanes))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

#resnet v1.5
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
# introduced by resnet end


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


class DepthWiseConv2d(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, kernel_size: int, stride: int, padding: int, groups: int, residual: bool=False):
        super(DepthWiseConv2d, self).__init__()
        self.conv1 = Conv2dBNPReLU(inplanes, groups, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2dBNPReLU(groups, groups, kernel_size, stride, padding, groups)
        self.conv3 = nn.Conv2d(groups, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.residual = residual
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        if self.residual:
            x = identity + x
        return x

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),        
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)        
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)            
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])        
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3        
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        out = [output1, output2, output3]
        return out

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)
