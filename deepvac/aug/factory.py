import sys
import re
import random
import cv2
from torchvision import transforms as trans
from ..utils import LOG
from ..core import AttrDict
from . import base_aug, face_aug, seg_aug, text_aug, yolo_aug

class SyszuxFactory(object):
    def __init__(self, syntax, deepvac_config):
        self.deepvac_config = deepvac_config
        self.deepvac_composer_config = deepvac_config.composer
        self.factory_dict = dict()
        self.initConfig()
        self.auditConfig()
        self.initSyntax(syntax)
        self.initProducts()

    def initConfig(self):
        if self.name() not in self.deepvac_composer_config.keys():
            self.deepvac_composer_config[self.name()] = AttrDict()
        self.config = self.deepvac_composer_config[self.name()]

    def name(self):
        return self.__class__.__name__

    def setAttr(self, k, v):
        self.config[k] = v

    def getAttr(self,k):
        return self.config[k]

    def auditConfig(self):
        pass

    def initSyntax(self, syntax):
        self.func_dict = dict()
        self.initChainKind()
        self.auditChainKind(syntax)
        flow = syntax.split(self.chain_kind)
        self.op_sym_list = [x.strip() for x in flow]
        self.p_list = [ float(x.split('@')[1].strip()) if '@' in x else 1 for x in self.op_sym_list  ]
        self.op_sym_list = [x.split('@')[0].strip() for x in self.op_sym_list if x]
        self.op_list = []

    def __call__(self, img):
        return self.func_dict[self.chain_kind](img)

    def addOp(self, op, p=1):
        self.op_list.append(op)
        self.p_list.append(p)

    def initChainKind(self):
        self.func_dict['=>'] = self.process

    def process(self, img):
        for t in self.op_list:
            img = t(img)
        return img

    def auditChainKind(self,flow_list):
        self.chain_kind = '=>'
        tokens = re.sub('[a-zA-Z0-9.@]'," ", flow_list).split()

        if len(tokens) == 0:
            return

        tokens = set(tokens)

        if len(tokens) > 1:
            raise Exception('Multi token found in flow list: ', tokens)

        self.chain_kind = tokens.pop()

        if self.chain_kind not in self.func_dict.keys():
            raise Exception("token not supported: ", self.chain_kind)

    def initProducts(self):
        LOG.logE("You must reimplement initProducts() function.", exit=True)

    def addProduct(self, name, ins):
        LOG.logE("You must reimplement addProduct() function in subclass.", exit=True)

    def initSym(self, module):
        return dir(module)

    def omitUnderScoreSym(self, sym):
        return [x for x in sym if not x.startswith('_')]

    def selectStartPrefix(self, sym, start_prefix):
        return [x for x in sym if x.startswith(start_prefix)]

    def selectEndSuffix(self, sym, end_suffix):
        return [x for x in sym if x.endswith(end_suffix)]

    def omitOmitList(self, sym, omit_list):
        return [x for x in sym if x not in omit_list]

    def getSymsFromProduct(self, module, start_prefix='', end_suffix='', omit_list=[]):
        sym = self.initSym(module)
        sym = self.omitUnderScoreSym(sym)
        sym = self.selectStartPrefix(sym, start_prefix)
        sym = self.selectEndSuffix(sym, end_suffix)
        return self.omitOmitList(sym, omit_list)

    def addProduct(self, sym, ins):
        self.factory_dict[sym] = ins

    def addProducts(self, module, module_name, start_prefix='', end_suffix='', sym_prefix='', omit_list=[]):
        for sym in self.getSymsFromProduct(module, start_prefix='', end_suffix='', omit_list=[]):
            self.addProduct(sym_prefix + sym, eval('{}.{}'.format(module_name, sym)) )

    def get(self, ins_name):
        if ins_name in self.factory_dict:
            return self.factory_dict[ins_name](self.deepvac_config)

        if ins_name in self.config:
            return self.config[ins_name]
        
        LOG.logE("ERROR! {} not found in factory {}.".format(ins_name, self.name()), exit=True)


class AugFactory(SyszuxFactory):
    def __init__(self, syntax, deepvac_config):
        super(AugFactory, self).__init__(syntax, deepvac_config)
        self.op_list = [self.get(x) for x in self.op_sym_list]

    def processRandom(self, img):
        for t,p in zip(self.op_list, self.p_list):
            if random.random() < p:
                img = t(img)
        return img

    def processDice(self, img):
        i = random.randrange(len(self.op_list))
        return self.op_list[i](img)

    def initChainKind(self):
        self.func_dict['=>'] = self.processRandom
        self.func_dict['||'] = self.processDice

    def initProducts(self):
        self.addProducts(base_aug, module_name='base_aug', end_suffix='Aug')
        self.addProducts(face_aug, module_name='face_aug', end_suffix='Aug')
        self.addProducts(seg_aug, module_name='seg_aug', end_suffix='Aug')
        self.addProducts(text_aug, module_name='text_aug', end_suffix='Aug')
        self.addProducts(yolo_aug, module_name='yolo_aug', end_suffix='Aug')
