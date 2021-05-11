import sys
from torchvision import transforms as trans
from ..utils import LOG
from ..core.factory import SyszuxFactory
from . import base_aug, face_aug, seg_aug, text_aug, yolo_aug

locals = locals()
class AugFactory(SyszuxFactory):
    def __init__(self):
        super(AugFactory,self).__init__()

    def auditConfig(self):
        super(AugFactory,self).auditConfig()
        self.end_suffix = 'Aug'
        self.locals = locals

    def addProduct(self, name, ins):
        self.factory_dict[name] = eval(ins)

    def initProducts(self):
        self.addProducts(base_aug, product_name='base_aug', end_suffix='Aug')
        self.addProducts(face_aug, product_name='face_aug', end_suffix='Aug')
        self.addProducts(seg_aug, product_name='seg_aug', end_suffix='Aug')
        self.addProducts(text_aug, product_name='text_aug', end_suffix='Aug')
        self.addProducts(yolo_aug, product_name='yolo_aug', end_suffix='Aug')
        self.addProducts(trans, product_name='trans', omit_list=["Compose","transforms","functional"])

    def initMyProducts(self):
        pass
        
if __name__ == "__main__":
    aug = AugFactory()
    LOG.logI(aug.factory_dict)
