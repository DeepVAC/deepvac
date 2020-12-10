import cv2
import numpy as np
import torch
import numpy as np
from torch import nn
from modules.model import model
from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_loader import OsWalkerLoader
from deepvac.syszux_report import ClassifierReport
from deepvac.syszux_aug import AugBase
from PIL import Image

class NSFWTestDataset(OsWalkerLoader):
    def __init__(self, nsfw_config):
        super(NSFWTestDataset, self).__init__(nsfw_config)

class DeepvacNSFW(Deepvac):
    def __init__(self, nsfw_config):
        super(DeepvacNSFW, self).__init__(nsfw_config)
        self.dataset = NSFWTestDataset(self.conf.test)

    def initNetWithCode(self):
        self.net = models.resnet50(pretrained=True).to(self.conf.device)

    def process(self):
        report = ClassifierReport(ds_name=self.conf.test.ds_name, cls_num=self.conf.cls_num)
        cls_to_idx = ['neutral', 'porn', 'sexy']

        for filename in self.dataset():
            self.target = filename.split('/')[-2]
            #if 4 channel, to 3 channels
            self.sample = Image.open(filename).convert('RGB')
            #self.sample = AugBase.cv2pillow(self.sample)
            self.sample = self.conf.test.transform_op(self.sample)
            self.sample = self.sample.unsqueeze(0).to(self.conf.device)
            # forward
            self.output = self.net(self.sample)
            # report
            gt = cls_to_idx.index(self.target)
            pred = self.output.argmax(1).item()
            report.add(gt, pred)
    
        report()

if __name__ == '__main__':
    from config import config 
    nsfw = DeepvacNSFW(config)
    nsfw()
