import torch.nn as nn

def initWeightsKaiming(civilnet):
    for m in civilnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def initWeightsNormal(civilnet):
    for m in civilnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)