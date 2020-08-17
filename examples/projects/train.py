import sys
sys.path.append('../deepvac/lib/')

import torch
from torch import nn
from torch import optim
from syszux_log import LOG
from modules.model import model
from torch.utils.data import DataLoader
from syszux_deepvac import DeepvacTrain
from torchvision.datasets import ImageFolder
from syszux_loader import ImageFolderWithTransformDataset


class NSFWTrainDataset(ImageFolderWithTransformDataset):
    def __init__(self, nsfw_config):
        super(NSFWTrainDataset, self).__init__(nsfw_config)

class NSFWValDataset(ImageFolderWithTransformDataset):
    def __init__(self, nsfw_config):
        super(NSFWValDataset, self).__init__(nsfw_config)

class DeepvacNSFW(DeepvacTrain):
    def __init__(self, nsfw_config):
        super(DeepvacNSFW, self).__init__(nsfw_config)

    def initNetWithCode(self):
        self.net = model.to(self.conf.device)

    def initModelPath(self):
        pass
    
    def initStateDict(self):
        pass

    def loadStateDict(self):
        pass

    def exportTorchViaScript(self):
        pass

    def initScheduler(self):
        pass

    def initCriterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def initTrainLoader(self):
        self.train_dataset = NSFWTrainDataset(self.conf.train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.conf.train.batch_size, num_workers=self.conf.num_workers, shuffle=self.conf.train.shuffle)

    def initValLoader(self):
        self.val_dataset = NSFWValDataset(self.conf.train)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.conf.val.batch_size, shuffle=self.conf.val.shuffle)
   
    def initOptimizer(self):
        self.initAdamOptimizer()

    def doForward(self):
        self.output = self.net(self.img)

    def doLoss(self):
        self.loss = self.criterion(self.output, self.idx)

    def doBackward(self):
        self.loss.backward()

    def doOptimize(self):
        self.optimizer.step()
 
    def preEpoch(self):
        if self.is_train:
            return
        self.accuracy_list = []

    def postIter(self):
        if self.is_train:
            return
        self.accuracy_list.append((self.output.argmax(-1) == self.idx).float().mean().item())

    def postEpoch(self):
        if self.is_train:
            return
        self.accuracy = sum(self.accuracy_list) / len(self.accuracy_list)


if __name__ == '__main__':
    from config import config 

    nsfw = DeepvacNSFW(config)
    nsfw("request")
