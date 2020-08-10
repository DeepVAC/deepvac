import os
from collections import OrderedDict
import sys
sys.path.append('lib')
from collections import defaultdict
from syszux_aug_factory import AugFactory
from syszux_loader_factory import LoaderFactory,DatasetFactory
from syszux_synthesis_factory import SynthesisFactory
import cv2
import random

class Chain(object):
    def __init__(self, flow_list):
        flow = flow_list.split("=>")
        self.op_sym_list = [x.strip() for x in flow]
        self.p_list = [ float(x.split('@')[1].strip()) if '@' in x else 1 for x in self.op_sym_list  ]
        self.op_sym_list = [x.split('@')[0].strip() for x in self.op_sym_list if x]
        self.op_list = []

    def __call__(self):
        for t in self.op_list:
            t()
    
    def addOp(self, op, p=1):
        self.op_list.append(op)
        self.p_list.append(p)

class ProductChain(Chain):
    def __init__(self, flow_list):
        super(ProductChain, self).__init__(flow_list)

    def __call__(self, img):
        for t in self.op_list:
            img = t(img)
        return img

class DeepvacChain(ProductChain):
    def __init__(self, flow_list, deepvac_config):
        super(DeepvacChain, self).__init__(flow_list)
        self.op_list = [eval("DeepvacChain.{}".format(x))(deepvac_config) for x in self.op_sym_list]

class AugChain(ProductChain):
    def __init__(self, flow_list, deepvac_config):
        super(AugChain, self).__init__(flow_list)
        self.factory = AugFactory()
        self.op_list = [self.factory.get(x)(deepvac_config) for x in self.op_sym_list]

class RandomAugChain(AugChain):
    def __call__(self, img):
        for t,p in zip(self.op_list, self.p_list):
            if random.random() < p:
                img = t(img)
        return img

class DiceAugChain(AugChain):
    def __call__(self, img):
        i = random.randrange(len(self.op_list))
        return self.op_list[i](img)
        
class Executor(object):
    def __init__(self,deepvac_config):
        self._graph = OrderedDict()
        self._graph_p = OrderedDict()
        self.conf = deepvac_config
        self.auditConfig()

    def addAugChain(self, name, chain, p=1.0):
        self._graph[name] = chain
        self._graph_p[name] = p

    def auditConfig(self):
        pass

    def addOp(self, name, op, p=1.0):
        self._graph[name].addOp(op,p)

    def remove(self, name):
        try:
            del self._graph[name]
        except KeyError:
            pass

    def __call__(self, img):
        for k in self._graph:
            if random.random() < self._graph_p[k]:
                img = self._graph[k](img)  
        return img   
              
class FaceAugExecutor(Executor):
    def __init__(self, deepvac_config):
        super(FaceAugExecutor, self).__init__(deepvac_config)

        ac1 = RandomAugChain('RandomColorJitterAug@0.5 => MosaicAug@0.5',deepvac_config)
        ac2 = DiceAugChain('IPCFaceAug => MotionAug',deepvac_config)

        self.addAugChain('ac1', ac1, 1)
        self.addAugChain('ac2', ac2, 0.5)
        
class AugExecutor(Executor):
    def __init__(self, deepvac_config):
        super(AugExecutor, self).__init__(deepvac_config)
        self.loader = LoaderFactory().get('OsWalkerLoader')(deepvac_config)
        self.aug_list = ['SpeckleAug','AffineAug','PerspectAug','GaussianAug','HorlineAug','VerlineAug','LRmotionAug','UDmotionAug','NoisyAug']
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
    x = Chain("RandomColorJitterAug@0.3 => MosaicAug@0.8 => MotionAug ")
    print(x.op_sym_list)
    print(x.p_list)
