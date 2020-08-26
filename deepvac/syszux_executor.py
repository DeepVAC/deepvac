import os
from collections import OrderedDict
import sys
sys.path.append('lib')
from collections import defaultdict
from .syszux_aug_factory import AugFactory
from .syszux_loader_factory import LoaderFactory,DatasetFactory
from .syszux_synthesis_factory import SynthesisFactory
from .syszux_log import LOG
import cv2
import re
import random

class Chain(object):
    def __init__(self, flow_list):
        self.func_dict = dict()
        self.initChainKind()
        self.auditChainKind(flow_list)
        flow = flow_list.split(self.chain_kind)
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

class AugChain(Chain):
    def __init__(self, flow_list, deepvac_config):
        super(AugChain, self).__init__(flow_list)
        self.factory = AugFactory()
        self.op_list = [self.factory.get(x)(deepvac_config) for x in self.op_sym_list]

    def initChainKind(self):
        self.func_dict['=>'] = self.processRandom
        self.func_dict['||'] = self.processDice

    def processRandom(self, img):
        for t,p in zip(self.op_list, self.p_list):
            if random.random() < p:
                img = t(img)
        return img

    def processDice(self, img):
        i = random.randrange(len(self.op_list))
        return self.op_list[i](img)

class DeepvacChain(Chain):
    def __init__(self, flow_list, deepvac_config):
        super(DeepvacChain, self).__init__(flow_list)
        self.op_list = [eval("DeepvacChain.{}".format(x))(deepvac_config) for x in self.op_sym_list]
        assert len(self.op_list) >0, 'module construct failed...'
        self.conf = deepvac_config

    def __call__(self, input):
        for o in self.op_list:
            o.setInput(input)
            o.process()
            input = o.getOutput()
        return input

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

        ac1 = AugChain('RandomColorJitterAug@0.5 => MosaicAug@0.5',deepvac_config)
        ac2 = AugChain('IPCFaceAug || MotionAug',deepvac_config)

        self.addAugChain('ac1', ac1, 1)
        self.addAugChain('ac2', ac2, 0.5)
        
class OcrAugExecutor(Executor):
    def __init__(self, deepvac_config):
        super(OcrAugExecutor, self).__init__(deepvac_config)
        self.loader = LoaderFactory().get('OsWalkerLoader')(deepvac_config)
        
        ac = AugChain('SpeckleAug || AffineAug || PerspectAug || GaussianAug || HorlineAug || VerlineAug || LRmotionAug || UDmotionAug || NoisyAug || DistortAug || PerspectiveAug || StretchAug',deepvac_config)
        self.addAugChain('ac', ac, self.conf.aug_rate)
        self.log_every = self.conf.log_every if self.conf.log_every!=None else 1000

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
        for idx, f in enumerate(self.loader()):
            img = cv2.imread(f)
            for k in self._graph:
                out = self._graph[k](img)
                self.dumpImgToPath(k, f, out)
            if idx % self.log_every == 0:
                LOG.logI("Current process: {}".format(idx))

if __name__ == "__main__":
    x = Chain("RandomColorJitterAug@0.3 => MosaicAug@0.8 => MotionAug ")
    print(x.op_sym_list)
    print(x.p_list)
    print(x.chain_kind)
