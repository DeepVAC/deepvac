import os
import sys
from datetime import datetime
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
import time
from enum import Enum
from syszux_log import LOG,getCurrentGitBranch

#deepvac implemented based on PyTorch Framework
class Deepvac(object):
    def __init__(self, deepvac_config):
        self._mandatory_member = dict()
        self._mandatory_member_name = ['']
        self.input_output = {'input':[], 'output':[]}
        self.conf = deepvac_config
        self.assertInGit()
        #init self.net
        self.initNet()

    def assertInGit(self):
        if os.environ.get("disable_git"):
            self.branch = "sevice"
            return

        if self.conf.disable_git:
            self.branch = "disable_git"
            return

        self.branch = getCurrentGitBranch()
        if self.branch is None:
            LOG.logE('According to deepvac standard, you must working in a git repo.', exit=True)
        
        if len(self.branch) < 6:
            LOG.logE('According to deepvac standard, your git branch name is too short: {}'.format(self.branch), exit=True)

        if self.branch.startswith('LTS_'):
            return
        
        if self.branch.startswith('PROTO_'):
            return

        LOG.logE('According to deepvac standard, git branch name should start from LTS_ or PROTO_: {}'.format(self.branch), exit=True)

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
                LOG.logE("Error! self.{} must be definded in your subclass.".format(name),exit=True)

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
        self.initLog()
        #init self.device
        self.initDevice()
        #init self.net
        self.initNetWithCode()
        self.initModelPath()
        #init self.model_dict
        self.initStateDict()
        #just load model after audit
        self.loadStateDict()
        self.exportTorchViaScript()

    def initDevice(self):
        #to determine CUDA device
        self.device = torch.device(self.conf.device)

    def initNetWithCode(self):
        raise Exception("You should reimplement this func to initialize self.net")

    def initModelPath(self):
        raise Exception("You should reimplement this func to initialize self.model_path")

    def initStateDict(self):
        LOG.logI('Loading State Dict from {}'.format(self.model_path))
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
        LOG.logI('Missing keys:{}'.format(len(missing_keys)))
        LOG.logI('Unused checkpoint keys:{}'.format(len(unused_keys)))
        LOG.logI('Used keys:{}'.format(len(used_keys)))
        assert len(used_keys) > 0, 'load NONE from pretrained model'
        assert len(missing_keys) == 0, 'Net mismatched with pretrained model'

    def loadStateDict(self):
        self.net.load_state_dict(self.state_dict, strict=False)
        self.net.eval()
        self.net = self.net.to(self.device)

    def initLog(self):
        pass

    def report(self):
        pass

    def process(self):
        pass

    def __call__(self,input):
        self.setInput(input)
        self.process()
        #post process
        if self.conf.script_model_dir:
            sys.exit(0)
        return self.getOutput()

    def exportTorchViaTrace(self, img):
        if not self.conf.trace_model_dir:
            return
        ts = torch.jit.trace(self.net, img)
        ts.save(self.conf.trace_model_dir)
        sys.exit(0)

    def exportTorchViaScript(self):
        if not self.conf.script_model_dir:
            return
        ts = torch.jit.script(self.net)
        ts.save(self.conf.script_model_dir)


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
        self.batch_size = self.conf.train.batch_size
        self.net.train()

    def setValContext(self):
        self.is_train = False
        self.is_val = True
        self.phase = 'VAL'
        self.dataset = self.val_dataset
        self.loader = self.val_loader
        self.batch_size = self.conf.val.batch_size
        self.net.eval()

    def initNet(self):
        super(DeepvacTrain,self).initNet()
        self.initOutputDir()
        self.initCriterion()
        self.initOptimizer()
        self.initScheduler()
        self.initCheckpoint()
        self.initTrainLoader()
        self.initValLoader()

    def initOutputDir(self):
        if self.conf.output_dir != 'output':
            LOG.logW("According deepvac standard, you should save model files to output directory.")

        self.output_dir = '{}/{}'.format(self.conf.output_dir, self.branch)
        LOG.logI('model save dir: {}'.format(self.output_dir))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def initCriterion(self):
        raise Exception("Not implemented.")

    def initCheckpoint(self, map_location=None):
        if not self.conf.checkpoint_suffix or self.conf.checkpoint_suffix == "":
            LOG.logI('Omit the checkpoint file since not specified...')
            return
        LOG.logI('Load checkpoint from {} folder'.format(self.output_dir))
        self.net.load_state_dict(torch.load(self.output_dir+'/model:{}'.format(conf.checkpoint_suffix), map_location=map_location))
        self.optimizer.load_state_dict(torch.load(self.output_dir+'/optimizer:{}'.format(conf.checkpoint_suffix), map_location=map_location))

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
        self.scheduler.step()

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
        torch.save(self.net.state_dict(), '{}/{}'.format(self.output_dir, self.state_file))
        torch.save(self.optimizer.state_dict(), '{}/{}'.format(self.output_dir, self.checkpoint_file))

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
            if i % self.conf.log_every == 0:
                LOG.logI('{}: [{}][{}/{}] [Loss:{}  Lr:{}]'.format(self.phase, self.epoch, self.step, loader_len,self.loss.item(),self.optimizer.param_groups[0]['lr']))
            self.postIter()
            if self.step in self.save_list:
                self.processVal()
                self.setTrainContext()
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
        assert self.train_sampler is not None, "You should define self.train_sampler in DDP mode."

    def initDDP(self):
        parser = argparse.ArgumentParser(description='DeepvacDDP')
        parser.add_argument("--gpu", default=-1, type=int, help="gpu")
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        self.args = parser.parse_args()
        self.map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.rank}

        LOG.logI("Start dist.init_process_group {} {}@{} on {}".format(self.conf.dist_url, self.args.rank, self.conf.world_size - 1, self.args.gpu))
        dist.init_process_group(backend='nccl', init_method=self.conf.dist_url, world_size=self.conf.world_size, rank=self.args.rank)
        torch.cuda.set_device(self.args.gpu)

    def initNet(self):
        self.initDDP()
        super(DeepvacDDP,self).initNet()

    def preEpoch(self):
        self.train_sampler.set_epoch(self.epoch)

    def saveState(self, time):
        if self.args.rank != 0:
            return
        super(DeepvacDDP, self).saveState(self.getTime())

    def initCheckpoint(self):
        super(DeepvacDDP, self).initCheckpoint(self.map_location)

    def separateBN4OptimizerPG(self, modules):
        paras_only_bn = []
        paras_wo_bn = []
        memo = set()
        gemfield_set = set()
        gemfield_set.update(set(modules.parameters()))
        LOG.logI("separateBN4OptimizerPG set len: {}".format(len(gemfield_set)))
        named_modules = modules.named_modules(prefix='')
        for module_prefix, module in named_modules:
            if "module" not in module_prefix:
                LOG.logI("separateBN4OptimizerPG skip {}".format(module_prefix))
                continue

            members = module._parameters.items()
            for k, v in members:
                name = module_prefix + ('.' if module_prefix else '') + k
                if v is None:
                    continue
                if v in memo:
                    continue
                memo.add(v)
                if "batchnorm" in str(module.__class__):
                    paras_only_bn.append(v)
                else:
                    paras_wo_bn.append(v)

        LOG.logI("separateBN4OptimizerPG param len: {} - {}".format(len(paras_wo_bn),len(paras_only_bn)))
        return paras_only_bn, paras_wo_bn

if __name__ == "__main__":
    from config import config as deepvac_config
    vac = Deepvac(deepvac_config)
    vac()
