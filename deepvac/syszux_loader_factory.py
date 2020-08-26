from torch.utils.data import Dataset,ConcatDataset,DataLoader

from .syszux_factory import SyszuxFactory
from . import syszux_loader

class DatasetFactory(SyszuxFactory):
    def __init__(self):
        super(DatasetFactory,self).__init__()

    def configure(self):
        self.end_suffix = 'Dataset'
        self.product_kind = syszux_loader

    def addProducts(self):
        for sym in self.getProducts():
            self.addProduct(sym, eval('{}.{}'.format('syszux_loader',sym)) )


class LoaderFactory(SyszuxFactory):
    def __init__(self):
        super(LoaderFactory,self).__init__()

    def configure(self):
        self.end_suffix = 'Loader'
        self.product_kind = syszux_loader

    def addProducts(self):
        for sym in self.getProducts():
            self.addProduct(sym, eval('{}.{}'.format('syszux_loader',sym)) )

if __name__ == "__main__":
    dataset = DatasetFactory()
    print(dataset.factory_dict)
    from torch.utils.data import DataLoader
    from config import config
    loader = DataLoader(dataset.get('FileLineDataset')(config), batch_size=1, shuffle=False, pin_memory=False, num_workers=1)
