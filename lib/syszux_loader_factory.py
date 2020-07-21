from torch.utils.data import Dataset,ConcatDataset,DataLoader

from factory import SyszuxFactory
import syszux_loader

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
    loader = LoaderFactory()
    print(loader.factory_dict)
    from torch.utils.data import DataLoader
    loader = DataLoader(loader['GemfieldLoader'], batch_size=1, shuffle=False, pin_memory=False, num_workers=1)