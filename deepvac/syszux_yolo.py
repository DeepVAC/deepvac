import torch.nn as nn
import torch.nn.modules.upsampling.Upsample as Upsample
from .syszux_modules import Conv2dBNHardswish, BottleneckStd, BottleneckCSP, SPP, Focus, Concat

class Detect(nn.Module):
    def __init__(self, class_num=80, anchors=(), in_planes_list=()):
        super(Detect, self).__init__()
        self.class_num = class_num
        self.output_num_per_anchor = class_num + 5 
        self.detect_layer_num = len(anchors)
        self.anchor_num = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.detect_layer_num
        anchor_t = torch.tensor(anchors).float().view(self.detect_layer_num, -1, 2)
        self.register_buffer('anchors', anchor_t)  # shape(detect_layer_num, anchor_num,2)
        self.register_buffer('anchor_grid', anchor_t.clone().view(self.detect_layer_num, 1, -1, 1, 1, 2))  # shape(detect_layer_num,1, anchor_num,1,1,2)
        self.conv_list = nn.ModuleList(nn.Conv2d(x, self.output_num_per_anchor * self.anchor_num, 1) for x in in_planes_list)

    def forward(self, x):
        inference_result = []  # inference output
        self.training |= self.export
        for i in range(self.detect_layer_num):
            x[i] = self.conv_list[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  
            #x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.anchor_num, self.output_num_per_anchor, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self.makeGrid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                inference_result.append(y.view(bs, -1, self.output_num_per_anchor))

        if self.training:
            return x
        
        return (torch.cat(inference_result, 1), x)

    @staticmethod
    def makeGrid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Yolo5(nn.Module):
    def __init__(self, class_num: int = 80):
        self.class_num = class_num
        self.upsample = Upsample(None, 2, nearest)
        self.cat = Concat(1)
        self.initBlock1()
        self.initBlock2()
        self.initBlock3()
        self.initBlock4()
        self.initBlock5()
        self.initDetect()

    def buildBlock(self, cfgs):
        layers = []
        #init the 4 layers
        for m, args in cfgs:
            layers.append(m(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x = self.upsample(x3)
        
        #cat from point2
        x = self.cat([x, x2])
        x4 = self.block4(x)
        x = self.upsample(x4)

        #cat from point1
        x = self.cat([x, x1])
        c1 = self.csp1(x)
        x = self.conv1(c1)

        #cat from point4
        x = self.cat([x,x4])
        c2 = self.csp2(x)
        x = self.conv2(x)

        #cat from point3
        x = self.cat([x,x3])
        c3 = self.csp3(x)

        return self.detect([c1,c2,c3])

    def initBlock1(self):
        cfgs = [
            [Focus, [3, 32, 3] ],
            [Conv2dBNHardswish, [32, 64, 3, 2] ],
            [BottleneckCSP, [64, 64, 1] ],
            [Conv2dBNHardswish, [64, 128, 3, 2] ],
            [BottleneckCSP, [128, 128, 3] ]
        ]
        self.block1 = self.buildBlock(cfgs)

    def initBlock2(self):
        cfgs = [
            [Conv2dBNHardswish, [128, 256, 3, 2] ],
            [BottleneckCSP, [256, 256, 3] ]
        ]
        self.block2 = self.buildBlock(cfgs)

    def initBlock3(self):
        cfgs = [
            [Conv2dBNHardswish, [256, 512, 3, 2] ],
            [SPP, [512, 512, [5, 9, 13]] ],
            [BottleneckCSP, [512, 512, 1, False] ],
            [Conv2dBNHardswish, [512, 256, 1, 1] ]
        ]
        self.block3 = self.buildBlock(cfgs)

    def initBlock4(self):
        cfgs = [
            [BottleneckCSP, [512, 256, 1, False] ],
            [Conv2dBNHardswish, [256, 128, 1, 1] ],
        ]
        self.block4 = self.buildBlock(cfgs)

    def initBlock5(self):
        self.csp1 = BottleneckCSP(256, 128, 1, False)
        self.conv1 = Conv2dBNHardswish(128, 128, 3, 2)
        self.csp1 = BottleneckCSP(256, 256, 1, False)
        self.conv2 = Conv2dBNHardswish(256, 256, 3, 2)
        self.csp3 = BottleneckCSP(512, 512, 1, False)

    def initDetect(self):
        #initial anchors
        self.detect = Detect(self.class_num, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512])