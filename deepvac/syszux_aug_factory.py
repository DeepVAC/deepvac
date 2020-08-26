try:
    from torchvision import transforms as trans
except:
    print("torchvision not found, some augmentors cannot be used.")

from .syszux_factory import SyszuxFactory
from . import syszux_aug

class AugFactory(SyszuxFactory):
    def __init__(self):
        super(AugFactory,self).__init__()

    def configure(self):
        self.start_prefix = ''
        self.omit_list = []
        self.product_kind = syszux_aug

    def addProducts(self):
        for sym in self.getProducts():
            self.addProduct(sym, eval('{}.{}'.format('syszux_aug',sym)) )
        
        self.addTVProducts()

    def configureTV(self):
        self.omit_list = ["Compose","transforms","functional"]
        self.product_kind = trans

    def addTVProducts(self):
        if 'trans' not in globals():
            return

        self.configureTV()
        #not the correct trans module
        if len(self.getProducts()) < 10:
            print("illegal torchvision found.")
            return

        for sym in self.getProducts():
            self.addProduct(sym, eval('{}.{}'.format('trans',sym)) )
        
if __name__ == "__main__":
    aug = AugFactory()
    print(aug.factory_dict)
