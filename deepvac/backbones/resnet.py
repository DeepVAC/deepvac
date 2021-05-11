import sys
import time
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as trans
from torchvision.models import resnet50
from ..core.config import config
from ..core.deepvac import Deepvac, DeepvacTrain
from ..datasets.os_walk import OsWalkDataset
from ..datasets.file_line import FileLineDataset
from ..utils import LOG
from .weights_init import initWeightsKaiming
from .bottleneck_layer import Bottleneck
from .conv_layer import Conv2dBNReLU

class ResnetBasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, outplanes: int, stride: int = 1):
        super(ResnetBasicBlock, self).__init__()
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

class ResNet18(nn.Module):
    def __init__(self, class_num: int = 1000):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.class_num = class_num
        self.auditConfig()
        self.conv1 = Conv2dBNReLU(in_planes=3, out_planes=self.inplanes, kernel_size=7, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        #init the 4 layers
        for outp, layer_num, stride in self.cfgs:
            layers.append(self.block(self.inplanes, outp, stride))
            self.inplanes = outp * self.block.expansion
            for _ in range(1, layer_num):
                layers.append(self.block(self.inplanes, outp))

        self.layer = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.initFc()
        initWeightsKaiming(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer(x)
        x = self.avgpool(x)
        return self.forward_cls(x)

    def forward_cls(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def initFc(self):
        self.fc = nn.Linear(512 * self.block.expansion, self.class_num)

    def auditConfig(self):
        self.block = ResnetBasicBlock
        self.cfgs = [
            # outp, layer_num, s
            [64,   2,  1],
            [128,  2,  2],
            [256,  2,  2],
            [512,  2,  2]
        ]

class ResNet34(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet34, self).__init__(class_num)

    def auditConfig(self):
        self.block = ResnetBasicBlock
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  4,  2],
            [256,  6,  2],
            [512,  3,  2]
        ]
    
class ResNet50(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet50, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  4,  2],
            [256,  6,  2],
            [512,  3,  2]
        ]

class ResNet101(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet101, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  4,  2],
            [256,  23,  2],
            [512,  3,  2]
        ]

class ResNet152(ResNet18):
    def __init__(self,class_num: int = 1000):
        super(ResNet152, self).__init__(class_num)

    def auditConfig(self):
        self.block = Bottleneck
        self.cfgs = [
            # outp, layer_num, s
            [64,   3,  1],
            [128,  8,  2],
            [256,  36,  2],
            [512,  3,  2]
        ]

class ResNet18OCR(ResNet18):
    def __init__(self):
        super(ResNet18OCR, self).__init__()

    def auditConfig(self):
        self.block = ResnetBasicBlock
        self.cfgs = [
            [64,   2,  1],
            [128,  2,  (2,1)],
            [256,  2,  (2,1)],
            [512,  2,  (2,1)]
        ]

    def initFc(self):
        self.avgpool = nn.AvgPool2d((2,2))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        x = self.avgpool(x)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2) # b *512 * width
        x = x.permute(2, 0, 1)  # [w, b, c]
        return x


class ResNet50Train(DeepvacTrain):
    def initNetWithCode(self):
        self.net = ResNet50()

    def initTrainLoader(self):
        self.train_dataset = FileLineDataset(self.conf.train)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.conf.train.batch_size, pin_memory=False)

    def initValLoader(self):
        self.val_dataset = FileLineDataset(self.conf.val)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, pin_memory=False)

    def postEpoch(self):
        if self.is_train:
            return
        self.accuracy = 1
        LOG.logI('Test accuray: {:.4f}'.format(self.accuracy))

class ResnetClsTestDataset(OsWalkDataset):
    def __init__(self, deepvac_config):
        super(ResnetClsTestDataset, self).__init__(deepvac_config)

    def __getitem__(self, index):
        filepath = self.files[index]
        sample = Image.open(filepath).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, filepath

class ResNet50Test(Deepvac):
    def __init__(self, deepvac_config):
        super(ResNet50Test, self).__init__(deepvac_config)

    def initNetWithCode(self):
        self.net = resnet50()
    
    def initTestLoader(self):
        if self.conf.disable_loader:
            return
        self.test_dataset = ResnetClsTestDataset(self.conf)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, pin_memory=False)

    def validate(self, t, img_path):
        LOG.logI('---------------VALIDATE BEGIN---------------')
        img = cv2.imread(img_path)
        if img.shape is None:
            LOG.logE('illegal image detected in validate!',exit=True)

        img = cv2.resize(img, (t.size(4), t.size(3)))
        transformer = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        input_t = transformer(img)
        input_t = input_t.to(self.device)

        torch.cuda.synchronize()
        start = time.time()

        preds = self.net(input_t.unsqueeze(0))
        torch.cuda.synchronize()
        end = time.time()
 
        softmaxs = F.softmax( preds, dim=1 )
        max_res = torch.max(softmaxs, dim=1)
        max_probability, max_index = max_res

        LOG.logI('Overall process time in validate: {} | Index: {} | Class: {} | Probability: {}'.format(end-start, max_index, max_index, max_probability))
        LOG.logI('---------------VALIDATE END---------------\n')

    def warmUp(self, t):
        LOG.logI('---------------WARMUP BEGIN---------------')

        for i in range(10):
            ti = torch.rand(( t.size(1), t.size(2), t.size(3), t.size(4) ), dtype = torch.float).to(self.device)
 
            torch.cuda.synchronize()
            start = time.time()

            resnet_out = self.net(ti)

            torch.cuda.synchronize()
            end = time.time()
        
            LOG.logI('Overall process time in warmup: {}'.format(end-start))

        LOG.logI('---------------WARMUP END---------------')

    def benchmark(self, t, img_path):
        self.validate(t, img_path)
        self.warmUp(t)

        item_num = t.size(0)
        LOG.logI('---------------BENCHMARK BEGIN--------------- {}'.format(item_num))

        torch.cuda.synchronize()
        start = time.time()

        for i in range(item_num):
            ti = t[i]
            torch.cuda.synchronize()
            tick = time.time()
            resnet_out = self.net(ti)
            
            torch.cuda.synchronize()
            tock = time.time()
            LOG.logI('forward:model forward time: {}'.format(tock-tick))

        torch.cuda.synchronize()
        end = time.time()

        print("|Model|Engine|Input size|forward time|")
        print("|-----|-------|----------|-----------|")
        print("|Resnet50|libtorch|{}x{}|{}|".format(t.size(4), t.size(3), (end-start)/item_num))
        LOG.logI('---------------BENCHMARK END---------------')

    def process(self):
        for input_tensor, path in self.test_loader:
            preds = self.net(input_tensor.to(self.device))
            softmaxs = F.softmax( preds, dim=1 )
            max_res = torch.max(softmaxs, dim=1)
            max_probability, max_index = max_res
            LOG.logI("path: {}, max_probability:{}, max_index:{}".format(path[0], max_probability.item(), max_index.item()))

def auditConfig():
    config.test.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.test.disable_git = True
    config.test.script_model_dir = "./gemfield_script.pt"
    config.test.transform = trans.Compose([
        trans.Resize((224, 224)),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    config.disable_git = True
    #train stuff
    config.momentum = 0.9
    config.weight_decay = 5e-4
    config.gamma = 0.1
    config.lr = 1e-3

    config.epoch_num = 100
    config.save_num = 1
    config.log_every = 100
    config.num_workers = 4
    config.lr_step = [50, 70, 90]
    config.lr_factor = 0.1

    config.train.shuffle = True
    config.train.batch_size = 1

    config.train.transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    config.val.transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        LOG.logE("Usage: python -m deepvac.syszux_resnet <train|test|benchmark> <pretrained_model.pth> <your_input>", exit=True)
    
    op = sys.argv[1]
    if op not in ('train','test','benchmark'):
        LOG.logE("Usage: python -m deepvac.syszux_resnet <train|test|benchmark> <pretrained_model.pth> <your_input>", exit=True)
    #
    auditConfig()

    if op == 'train':
        if(len(sys.argv) != 6):
            LOG.logE("Usage: python -m deepvac.syszux_resnet train <pretrained_model.pth> <train_val_data_dir_prefix> <train.txt> <val.txt>", exit=True)

        config.model_path = sys.argv[2]
        config.train.fileline_data_path_prefix = sys.argv[3]
        config.train.fileline_path = sys.argv[4]
        config.val.fileline_data_path_prefix = sys.argv[3]
        config.val.fileline_path = sys.argv[5]
        train = ResNet50Train(config)
        train()

    if op == 'test':
        if(len(sys.argv) != 4):
            LOG.logE("Usage: python -m deepvac.syszux_resnet test <pretrained_model.pth> <your_test_img_input_dir>", exit=True)

        config.test.model_path = sys.argv[2]
        config.test.static_quantize_dir = "./static_quantize.pt"
        config.test.input_dir = sys.argv[3]
        test = ResNet50Test(config.test)
        test()

    if op == 'benchmark':
        if(len(sys.argv) != 4):
            LOG.logE("Usage: python -m deepvac.syszux_resnet benchmark <pretrained_model.pth> <your_input_img.jpg>", exit=True)

        config.test.model_path = sys.argv[2]
        config.test.disable_loader = True
        img_path = sys.argv[3]

        t224x224 = torch.rand((100, 1, 3, 224, 224), dtype = torch.float).to(config.test.device)
        t640x640 = torch.rand((100, 1, 3, 640, 640), dtype = torch.float).to(config.test.device)
        t1280x720 = torch.rand((50, 1, 3, 720, 1280), dtype = torch.float).to(config.test.device)
        t1280x1280 = torch.rand((50, 1, 3, 1280, 1280), dtype = torch.float).to(config.test.device)

        test = ResNet50Test(config.test)
        test.benchmark(t224x224, img_path)
        test.benchmark(t640x640, img_path)
        test.benchmark(t1280x720, img_path)
        test.benchmark(t1280x1280, img_path)