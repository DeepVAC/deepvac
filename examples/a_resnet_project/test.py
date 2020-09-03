import cv2
import torch
import numpy as np
from torch import nn
from modules.model import model
from deepvac.syszux_log import LOG
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_loader import OsWalkerLoader
from deepvac.syszux_report import ClassifierReport

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

    def process(self):
        self.initNet()
        for filename in self.dataset():
            # label
            label = filename.split('/')[-2]
            # img
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            img = cv2.resize(img, self.conf.test.input_size, interpolation=cv2.INTER_LINEAR)
            img = (img / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
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
