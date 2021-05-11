import numpy as np
import torch
import torch.nn as nn
from ..utils import LOG
from .weights_init import initWeightsKaiming
from .conv_layer import Conv2dBNWithName

class RepVGGModelConvert(object):
    def __init__(self):
        self.all_weights = {}

    def _isValidName(self, name):
        if name[0] == '.':
            return False
        if '.rbr_dense' in name:
            return False
        if '.rbr_1x1' in name:
            return False
        if '.rbr_identity' in name:
            return False
        return True

    def _addReparamPart(self, name, module):
        kernel, bias = module.repvgg_convert()
        self.all_weights[name + '.rbr_reparam.weight'] = kernel
        self. all_weights[name + '.rbr_reparam.bias'] = bias

    def _addOtherPart(self, name, module):
        for p_name, p_tensor in module.named_parameters():
            full_name = name + '.' + p_name
            if self._isValidName(full_name) and full_name not in self.all_weights:
                self.all_weights[full_name] = p_tensor.detach()
        for p_name, p_tensor in module.named_buffers():
            full_name = name + '.' + p_name
            if self._isValidName(full_name) and full_name not in self.all_weights:
                self.all_weights[full_name] = p_tensor

    def __call__(self, train_model:torch.nn.Module, deploy_model:torch.nn.Module, save_path=None):
        self.all_weights = {}
        for name, module in train_model.named_modules():
            if hasattr(module, 'repvgg_convert'):
                self._addReparamPart(name, module)
            else:
                self._addOtherPart(name, module)

        deploy_model.load_state_dict(self.all_weights)
        if save_path is not None:
            torch.save(deploy_model.state_dict(), save_path)

        return deploy_model

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        kernel_size = 3
        padding = 1

        padding_11 = padding - kernel_size // 2

        self.relu = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = Conv2dBNWithName(in_channels, out_channels, kernel_size, stride, padding, groups)
            self.rbr_1x1 = Conv2dBNWithName(in_channels, out_channels, 1, stride, padding_11, groups)

    def forward(self, inputs):
        if self.deploy:
            return self.relu(self.rbr_reparam(inputs))

        id_out = self.rbr_identity(inputs) if self.rbr_identity else 0

        return self.relu(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel_1x1):
        if kernel_1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel_1x1, [1,1,1,1])

    def _fuse_bn_tensor_for_sequential(self, branch):
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor_for_bn(self, branch):
        if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
        kernel = self.id_tensor
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        
        if isinstance(branch, nn.Sequential):
            return self._fuse_bn_tensor_for_sequential(branch)
        elif isinstance(branch, nn.BatchNorm2d):
            return self._fuse_bn_tensor_for_bn(branch)

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach(), bias.detach()

class RepVGG(nn.Module):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGG, self).__init__()

        self.deploy = deploy
        self.auditConfig()

        self.in_planes = min(64, self.cfgs[0][0])
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, stride=2, deploy=self.deploy)
        self.cur_layer_idx = 1

        layers = []
        for planes, num_blocks, stride in self.cfgs:
            layers.extend(self._make_layer(planes, num_blocks, stride))
        
        self.layer = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(self.cfgs[-1][0], class_num)

        initWeightsKaiming(self)

    def auditConfig(self):
        self.cfgs = None
        self.override_groups_map = None
        LOG.logE("You must reimplement auditConfig() to initialize self.cfgs and self.override_groups_map", exit=True)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, stride=stride, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return blocks

    def forward(self, x):
        out = self.stage0(x)
        out = self.layer(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class RepVGGASmall(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGASmall, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [48, 2, 2],
            [96, 4, 2],
            [192, 14, 2],
            [1280, 1, 2]
        ]
        self.override_groups_map = dict()

class RepVGGAMedium(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGAMedium, self).__init__(class_num, deploy)
    
    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [64, 2, 2],
            [128, 4, 2],
            [256, 14, 2],
            [1280, 1, 2]
        ]
        self.override_groups_map = dict()

class RepVGGALarge(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGALarge, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [96, 2, 2],
            [192, 4, 2],
            [384, 14, 2],
            [1408, 1, 2]
        ]
        self.override_groups_map = dict()

class RepVGGBSmall(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBSmall, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [64, 4, 2],
            [128, 6, 2],
            [256, 16, 2],
            [1280, 1, 2]
        ]
        self.override_groups_map = dict()

class RepVGGBMedium(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBMedium, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [128, 4, 2],
            [256, 6, 2],
            [512, 16, 2],
            [2048, 1, 2]
        ]
        self.override_groups_map = dict()

class RepVGGBLarge(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBLarge, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [160, 4, 2],
            [320, 6, 2],
            [640, 16, 2],
            [2560, 1, 2]
        ]
        self.override_groups_map = dict()

class RepVGGBMediumG2(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBMediumG2, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [128, 4, 2],
            [256, 6, 2],
            [512, 16, 2],
            [2048, 1, 2]
        ]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: 2 for l in optional_groupwise_layers}

class RepVGGBMediumG4(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBMediumG4, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [128, 4, 2],
            [256, 6, 2],
            [512, 16, 2],
            [2048, 1, 2]
        ]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: 4 for l in optional_groupwise_layers}

class RepVGGBLargeG2(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBLargeG2, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [160, 4, 2],
            [320, 6, 2],
            [640, 16, 2],
            [2560, 1, 2]
        ]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: 2 for l in optional_groupwise_layers}

class RepVGGBLargeG4(RepVGG):
    def __init__(self, class_num=1000, deploy=False):
        super(RepVGGBLargeG4, self).__init__(class_num, deploy)

    def auditConfig(self):
        self.cfgs = [
            # planes, num_blocks, stride
            [160, 4, 2],
            [320, 6, 2],
            [640, 16, 2],
            [2560, 1, 2]
        ]
        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.override_groups_map = {l: 4 for l in optional_groupwise_layers}

if __name__ == '__main__':
    model = RepVGGASmall()
    LOG.logI(model)
    x = torch.randn((1,3,32,320))
    LOG.logI(model(x).shape)
