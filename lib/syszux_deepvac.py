import os
import sys
from datetime import datetime
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
import time
from enum import Enum
from syszux_log import LOG

#deepvac implemented based on PyTorch Framework
class Deepvac(object):
    def __init__(self, deepvac_config):
        self._mandatory_member = dict()
        self._mandatory_member_name = ['']
        self.input_output = {'input':[], 'output':[]}
        self.conf = deepvac_config
        #init self.net
        self.initNet()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        self.checkIn(name)

    def __delattr__(self, name):
        self.checkOut(name)
        object.__delattr__(self, name)

    def checkIn(self, name):
        if name.startswith('_'):
            return
        self._mandatory_member[name] = None

    def checkOut(self, name):
        if name.startswith('_'):
            return
        del self._mandatory_member[name]

    def auditConfig(self):
        for name in self._mandatory_member_name:
            if name not in self._mandatory_member:
                LOG.logE("Error! self.{} must be definded in your subclass.".format(name))
                sys.exit(1)

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

    def getTime(self):
        return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

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

    def initDevice(self):
        #to determine CUDA device
        self.device = torch.device(self.conf.device)

    def initNetWithCode(self):
        raise Exception("You should reimplement this func to initialize self.net")

    def initModelPath(self):
        raise Exception("You should reimplement this func to initialize self.model_path")

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

    def __call__(self,input):
        self.setInput(input)
        self.process()
        return self.getOutput()

class DeepvacTrain(Deepvac):
    #net must be PyTorch Module.
    def __init__(self, deepvac_config):
        super(DeepvacTrain,self).__init__(deepvac_config)
        self.dataset = None
        self.loader = None
        self.epoch = 0
        self.step = 0
        self.iter = 0
        self._mandatory_member_name = ['train_dataset','val_dataset','train_loader','val_loader','net','criterion','optimizer']

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

    def initNet(self):
        super(DeepvacTrain,self).initNet()
        self.initCriterion()
        self.initOptimizer()
        self.initCheckpoint()
        self.initScheduler()
        self.initTrainLoader()
        self.initValLoader()
        self.initOutputDir()

    def initOutputDir(self):
        print('debug output: ',self.conf.output_dir)
        if not os.path.exists(self.conf.output_dir):
            os.makedirs(self.conf.output_dir)

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

    def preEpoch(self):
        pass

    def preIter(self):
        pass

    def postIter(self):
        pass

    def postEpoch(self):
        pass

    def doForward(self):
        raise Exception('Not implemented.')

    def doLoss(self):
        raise Exception('Not implemented.')

    def doBackward(self):
        raise Exception('Not implemented.')

    def doOptimize(self):
        raise Exception('Not implemented.')

    def saveState(self, time):
        self.state_file = 'model:{}_acc:{}_epoch:{}_step:{}_lr:{}.pth'.format(time, self.accuracy, self.epoch, self.step, self.optimizer.param_groups[0]['lr'])
        self.checkpoint_file = 'optimizer:{}_acc:{}_epoch:{}_step:{}_lr:{}.pth'.format(time, self.accuracy, self.epoch, self.step, self.optimizer.param_groups[0]['lr'])
        torch.save(self.net.state_dict(), '{}/{}'.format(self.conf.output_dir, self.state_file))
        torch.save(self.optimizer.state_dict(), '{}/{}'.format(self.conf.output_dir, self.checkpoint_file))

    def processTrain(self):
        self.setTrainContext()
        self.step = 0
        LOG.logI('Phase {} started...'.format(self.phase))
        self.preEpoch()
        loader_len = len(self.loader)
        save_every = loader_len//self.conf.save_num
        save_list = list(range(0,loader_len + 1, save_every ))
        self.save_list = save_list[1:-1]
        LOG.logI('SAVE LIST: {}'.format(self.save_list))

        for i, (img, idx) in enumerate(self.loader):
            self.step = i
            self.iter += 1
            self.idx = idx
            self.img = img
            self.preIter()
            self.doForward()
            self.doLoss()
            self.doBackward()
            self.doOptimize()
            LOG.logI('{}: [{}][{}/{}] [Loss:{}  Lr:{}]'.format(self.phase, self.epoch, self.step, loader_len,self.loss.item(),self.optimizer.param_groups[0]['lr']))
            self.postIter()
            if self.step in self.save_list:
                self.processVal()

        self.scheduler.step()
        self.postEpoch()

    def processVal(self):
        self.setValContext()
        LOG.logI('Phase {} started...'.format(self.phase))
        with torch.no_grad():
            self.preEpoch()
            for i, (img, idx) in enumerate(self.loader):
                self.idx = idx
                self.img = img
                self.preIter()
                self.doForward()
                self.doLoss()
                LOG.logI('{}: [{}][{}/{}]'.format(self.phase, self.epoch, i, len(self.loader)))
                self.postIter()

            self.postEpoch()
        self.saveState(self.getTime())
        

    def processAccept(self):
        self.setValContext()

    def process(self):
        self.iter = 0
        for epoch in range(self.conf.epoch_num):
            self.epoch = epoch
            LOG.logI('Epoch {} started...'.format(self.epoch))
            self.processTrain()
            self.processVal()
            self.processAccept()

    def __call__(self,input):
        self.auditConfig()
        self.process()

    def exportNCNN(self):
        pass

    def exportCoreML(self):
        pass

    def exportONNX(self):
        pass

class DeepvacDDP(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(DeepvacDDP,self).__init__(deepvac_config)

    def initDDP(self):
        parser = argparse.ArgumentParser(description='DeepvacDDP')
        parser.add_argument("--gpu", default=-1, type=int, help="gpu")
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        self.args = parser.parse_args()
        self.map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.rank}

        LOG.logI("Start dist.init_process_group {} {}@{} on {}".format(self.conf.dist_url, self.args.rank, self.conf.world_size - 1, self.args.gpu))
        dist.init_process_group(backend='nccl', init_method=self.conf.dist_url, world_size=self.conf.world_size, rank=self.args.rank)
        torch.cuda.set_device(self.args.gpu)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.args.gpu])
        LOG.logI("Finish dist.init_process_group {} {}@{} on {}".format(self.conf.dist_url, self.args.rank, self.conf.world_size - 1, self.args.gpu))

    def initNet(self):
        super(DeepvacDDP,self).initNet()
        self.initDDP()

    def saveState(self, time):
        if self.args.rank != 0:
            return
        super(DeepvacDDP, self).saveState(self.getTime())

    def loadState(self, suffix):
        self.optimizer.load_state_dict(torch.load(self.conf.output_dir/'optimizer:{}'.format(suffix), map_location=self.map_location))
        self.model.load_state_dict(torch.load(self.conf.output_dir/'model:{}'.format(suffix),map_location=self.map_location))

if __name__ == "__main__":
    from conf import config as deepvac_config
    vac = Deepvac(deepvac_config)
    vac()
