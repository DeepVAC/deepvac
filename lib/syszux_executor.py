import os
from collections import OrderedDict
import sys
sys.path.append('lib')
from collections import defaultdict
from syszux_aug_factory import AugFactory
from syszux_loader_factory import LoaderFactory
from syszux_synthesis_factory import SynthesisFactory
import cv2

class Chain(object):
    def __init__(self, flow_list):
        flow = flow_list.split("=>")
        self.op_sym_list = [x.strip() for x in flow]
        self.op_sym_list = [x.strip() for x in self.op_sym_list if x]
        self.op_list = []

    def __call__(self):
        for t in self.op_list:
            t()
    
    def addOp(self, op):
        self.op_list.append(op)

class ProductChain(Chain):
    def __init__(self, flow_list):
        super(ProductChain, self).__init__(flow_list)

    def __call__(self, img):
        for t in self.op_list:
            img = t(img)
        return img
    
    def addOp(self, op):
        self.op_list.append(op)

class DeepvacChain(ProductChain):
    def __init__(self, flow_list, deepvac_config):
        super(DeepvacChain, self).__init__(flow_list)
        self.op_list = [eval("DeepvacChain.{}".format(x))(deepvac_config) for x in self.op_sym_list]

class AugChain(ProductChain):
    def __init__(self, flow_list, deepvac_config):
        super(AugChain, self).__init__(flow_list)
        self.factory = AugFactory()
        self.op_list = [self.factory.get(x)(deepvac_config) for x in self.op_sym_list]
        
class Executor(object):
    def __init__(self,deepvac_config):
        self._graph = OrderedDict()
        self.loader = None
        self.conf = deepvac_config
        self.auditConfig()

    def addAugChain(self, name, chain):
        self._graph[name] = chain

    def auditConfig(self):
        pass

    def addOp(self, name, op):
        self._graph[name].addOp(op)

    def remove(self, name):
        try:
            del self._graph[name]
        except KeyError:
            pass

    def __call__(self):
        for f in self.loader:
            img = cv2.imread(f)
            for k in self._graph:
                x = self._graph[k](img)  

class AugExecutor(Executor):
    def __init__(self, deepvac_config):
        super(AugExecutor, self).__init__(deepvac_config)
        self.loader = LoaderFactory().get('OsWalkerLoader')(deepvac_config)
        self.aug_list = ['SpeckleAug','AffineAug','PerspectAug','GaussianAug','HorlineAug','VerlineAug','LRmotionAug','UDmotionAug','NoisyAug']
        self.aug_list = ['SpeckleAug']
        aug_factory = AugFactory()
        for a in self.aug_list:
            self.addAugChain(a, AugChain(a,deepvac_config))
            # self.addOp(a, aug_factory.get(a))

    def auditConfig(self):
        self.output_dir = self.conf.output_dir

    def dumpImgToPath(self, aug_kind, file_name, img):
        fdir,fname = file_name.split(os.sep)[-2:]

        #make sure output dir exist
        output_dir = os.path.join(self.conf.output_dir, fdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #replace name
        fname = '{}_{}'.format(aug_kind, fname)
        output_file_name = os.path.join(output_dir, fname)
        cv2.imwrite(output_file_name, img)


    def __call__(self):
        for f in self.loader():
            img = cv2.imread(f)
            for k in self._graph:
                out = self._graph[k](img)
                # print(k,f, type(out))
                self.dumpImgToPath(k, f, out)

if __name__ == "__main__":
    from config import config as deepvac_config
    executor = AugExecutor(deepvac_config)
    executor()
    