from .syszux_factory import SyszuxFactory
import .syszux_synthesis

class SynthesisFactory(SyszuxFactory):
    def __init__(self):
        super(SynthesisFactory,self).__init__()

    def configure(self):
        self.start_prefix = 'SynthesisText'
        self.omit_list = []
        self.product_kind = syszux_synthesis

    def addProducts(self):
        for sym in self.getProducts():
            self.addProduct(sym, eval('{}.{}'.format('syszux_synthesis',sym)) )
        
if __name__ == "__main__":
    synthesis = SynthesisFactory()
    print(synthesis.factory_dict)