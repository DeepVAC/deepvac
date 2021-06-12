from torch.utils.data import Dataset
from ..utils import LOG, addUserConfig
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

    def auditConfig(self):
        pass

    def addUserConfig(self, config_name, user_give=None, developer_give=None, is_user_mandatory=False):
        module_name = 'config.datasets.{}'.format(self.name())
        return addUserConfig(module_name, config_name, user_give=user_give, developer_give=developer_give, is_user_mandatory=is_user_mandatory)

    def compose(self, img):
        if isinstance(self.config.transform, (list,tuple)):
            for t in self.config.transform:
                img = t(img)
        elif self.config.transform is not None:
            img = self.config.transform(img)

        if isinstance(self.config.composer, (list,tuple)):
            for c in self.config.composer:
                img = c(img)
        elif self.config.composer is not None:
            img = self.config.composer(img)

        return img

    def name(self):
        return self.__class__.__name__

    def setAttr(self, k, v):
        self.config[k] = v

    def getAttr(self,k):
        return self.config[k]
