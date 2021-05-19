from torch.utils.data import Dataset
from ..utils import LOG
from ..core import AttrDict

class DatasetBase(Dataset):
    def __init__(self, deepvac_config):
        super(DatasetBase, self).__init__()
        self.deepvac_datasets_config = deepvac_config.datasets
        self.initConfig()
        self.auditConfig()

    def initConfig(self):
        if self.name() not in self.deepvac_datasets_config.keys():
            self.deepvac_datasets_config[self.name()] = AttrDict()
        self.config = self.deepvac_datasets_config[self.name()]
        self.transform = self.config.transform
        self.composer = self.config.composer

    def auditConfig(self):
        pass

    def name(self):
        return self.__class__.__name__

    def setAttr(self, k, v):
        self.config[k] = v

    def getAttr(self,k):
        return self.config[k]
