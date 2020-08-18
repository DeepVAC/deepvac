import sys
sys.path.append('../../lib/')

import cv2
import torch

from torch import nn
from syszux_log import LOG
from modules.model import model
from syszux_deepvac import Deepvac
from syszux_loader import OsWalkerLoader
from syszux_report import ClassifierReport


class NSFWTestDataset(OsWalkerLoader):
    def __init__(self, nsfw_config):
        super(NSFWTestDataset, self).__init__(nsfw_config)

class DeepvacNSFW(Deepvac):
    def __init__(self, nsfw_config):
        super(DeepvacNSFW, self).__init__(nsfw_config)
        self.dataset = NSFWTestDataset(self.conf.test)
        self.report = ClassifierReport(ds_name=self.conf.test.ds_name, cls_num=self.conf.cls_num)

    def initNetWithCode(self):
        self.net = model.to(self.conf.device)

    def initModelPath(self):
        self.model_path = self.conf.test.model_path

    def process(self):
        self.initNet()
        for filename in self.dataset():
            # label
            label = filename.split('/')[-2]
            # img
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.conf.device)
            # forward
            with torch.no_grad():
                self.output = self.net(img)
            # report
            gt = self.conf.test.cls_to_idx.index(label)
            pred = self.output.argmax(1).item()
            self.report.add(gt, pred)
    
    def __call__(self):
        self.process()
        self.report()


if __name__ == '__main__':
    from config import config 

    nsfw = DeepvacNSFW(config)
    nsfw()
