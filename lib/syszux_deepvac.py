import os
import torch
import torch.optim as optim
import time
from enum import Enum
from syszux_log import LOG

#deepvac implemented based on PyTorch Framework
class DeepVAC(object):
    class STATE(Enum):
        OK = 'OK'
        ERROR = 'ERROR'
        LOAD_NET = 'LOAD'
        LOAD_DB = "LOAD_DB"
        RELOAD_DB = "RELOAD_DB"

    #net must be PyTorch Module.
    def __init__(self, deepvac_config):
        self.state_dict = {'RELOAD_DB':DeepVAC.STATE.RELOAD_DB}
        self.input_output = {'input':[], 'output':[]}
        self.conf = deepvac_config
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.loader = None
        self.train_loader = None
        self.val_loader = None
        self.batch_size = None
        self.phase = 'None'
        #init self.net
        self.initNet()

    def setTrainContext(self):
        self.is_train = True
        self.is_val = False
        self.phase = 'TRAIN'
        self.dataset = self.train_dataset
        self.loader = self.train_loader
        self.batch_size = self.conf.train_batch_size
        self.net.train()

    def setValContext(self):
        self.is_train = False
        self.is_val = True
        self.phase = 'VAL'
        self.dataset = self.val_dataset
        self.loader = self.val_loader
        self.batch_size = self.conf.val_batch_size
        self.net.eval()

    def getConf(self):
        return self.conf

    def setInput(self, input):
        if not isinstance(input, list):
            input = [input]
 
        self.input_output['input'].extend(input)
        self.input_output['output'].clear()

    def addOutput(self, output):
        if not isinstance(output, list):
            output = [output]
        self.input_output['output'].extend(output)

    def getOutput(self):
        self.input_output['input'].clear()
        return self.input_output['output']

    def initNet(self):
        #init self.device
        self.initDevice()
        #init self.net
        self.initNetWithCode()
        self.initModelPath()
        #init self.model_dict
        self.initStateDict()
        #just load model after audit
        self.loadStateDict()
        self.initCriterion()
        self.initOptimizer()
        self.initCheckpoint()
        self.initScheduler()
        self.initTrainLoader()
        self.initValLoader()
        self.initOutputDir()
        self.initDDP()

    def initOutputDir(self):
        pass

    def initDevice(self):
        #to determine CUDA device
        self.device = torch.device(self.conf.device)

    def initNetWithCode(self):
        raise Exception("You should reimplement this func to initialize self.net")

    def initModelPath(self):
        raise Exception("You should reimplement this func to initialize self.model_path")

    def initCriterion(self):
        raise Exception("Not implemented.")

    def initCheckpoint(self):
        raise Exception("Not implemented.")

    def initScheduler(self):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.conf.lr_step,self.conf.lr_factor)

    def initTrainLoader(self):
        raise Exception("Not implemented.")

    def initValLoader(self):
        raise Exception("Not implemented.")

    def initDDP(self):
        raise Exception("Not implemented.")

    def initStateDict(self):
        LOG.log(LOG.S.I, 'Loading State Dict from {}'.format(self.model_path))
        device = torch.cuda.current_device()
        self.state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage.cuda(device))
        #remove prefix begin
        prefix = 'module.'
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        if "state_dict" in self.state_dict.keys():
            self.state_dict = {f(key): value for key, value in self.state_dict['state_dict'].items()}
        else:
            self.state_dict = {f(key): value for key, value in self.state_dict.items()}
        #remove prefix end

        # just do audit on model file
        state_dict_keys = set(self.state_dict.keys())
        code_net_keys = set(self.net.state_dict().keys())
        used_keys = code_net_keys & state_dict_keys
        unused_keys = state_dict_keys - code_net_keys
        missing_keys = code_net_keys - state_dict_keys
        LOG.log(LOG.S.I, 'Missing keys:{}'.format(len(missing_keys)))
        LOG.log(LOG.S.I, 'Unused checkpoint keys:{}'.format(len(unused_keys)))
        LOG.log(LOG.S.I, 'Used keys:{}'.format(len(used_keys)))
        assert len(used_keys) > 0, 'load NONE from pretrained model'
        assert len(missing_keys) == 0, 'Net mismatched with pretrained model'

    def loadStateDict(self):
        self.net.load_state_dict(self.state_dict, strict=False)
        self.net.eval()
        self.net = self.net.to(self.device)

    def report(self):
        pass

    def initOptimizer(self):
        self.initSgdOptimizer()

    def initSgdOptimizer(self):
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.conf.lr,
            momentum=self.conf.momentum,
            weight_decay=self.conf.weight_decay,
            nesterov=self.conf.nesterov
        )

    def initAdamOptimizer(self):
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.conf.lr,
        )
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    
    def initRmspropOptimizer(self):
        self.optimizer = optim.RMSprop(
            self.net.parameters(),
            lr=self.conf.lr,
            momentum=self.conf.momentum,
            weight_decay=self.conf.weight_decay,
            # alpha=self.conf.rmsprop_alpha,
            # centered=self.conf.rmsprop_centered
        )

    def preEpoch(self, epoch=0):
        pass

    def preIter(self, img=None, idx=0, epoch=0):
        pass

    def postIter(self, img=None, idx=0, epoch=0):
        pass

    def postEpoch(self, epoch=0):
        pass

    def doForward(self):
        raise Exception('Not implemented.')

    def doLoss(self):
        raise Exception('Not implemented.')

    def doBackward(self):
        raise Exception('Not implemented.')

    def doOptimize(self):
        raise Exception('Not implemented.')

    def processTrain(self, epoch):
        self.setTrainContext()
        LOG.logI('Phase {} started...'.format(self.phase))
        self.preEpoch(epoch)
        for i, (img, idx) in enumerate(self.loader):
            self.preIter(img, idx)
            self.doForward(idx)
            self.doLoss(idx)
            self.doBackward(idx)
            self.doOptimize(idx)
            LOG.logI('{}: [{}][{}/{}] [Loss:{}  Lr:{}]'.format(self.phase, epoch, i, len(self.loader),self.loss.item(),self.optimizer.param_groups[0]['lr']))
            self.postIter(img, idx)

        self.lr_scheduler.step()
        self.postEpoch(epoch)

    def processVal(self, epoch):
        self.setValContext()
        LOG.logI('Phase {} started...'.format(self.phase))
        with torch.no_grad():
            self.preEpoch(epoch)
            for i, (img, idx) in enumerate(self.loader):
                self.preIter(img, idx)
                self.doForward(idx)
                self.doLoss(idx)
                LOG.logI('{}: [{}][{}/{}]'.format(self.phase, epoch, i, len(self.loader)))
                self.postIter(img, idx)

            self.postEpoch(epoch)

    def processAccept(self, epoch):
        self.setValContext()

    def process(self):
        for epoch in range(self.conf.epoch_num):
            LOG.logI('Epoch {} started...'.format(epoch))
            self.processTrain(epoch)
            self.processVal(epoch)
            self.processAccept(epoch)

    def __call__(self,input):
        self.setInput(input)
        self.process()
        return self.getOutput()

    def exportNCNN(self):
        pass

    def exportCoreML(self):
        pass

    def exportONNX(self):
        pass

if __name__ == "__main__":
    from conf import config as deepvac_config
    vac = DeepVAC(deepvac_config)
    vac()
