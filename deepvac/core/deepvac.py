import os
import argparse
import collections
import time
import copy
import math
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    LOG.logE("Deepvac has dependency on tensorboard, please install tensorboard first, e.g. [pip3 install tensorboard]", exit=True)

from ..utils import syszux_once, LOG, assertAndGetGitBranch, getPrintTime, AverageMeter, anyFieldsInConfig, addUserConfig
from ..cast import export3rd
from .config import AttrDict

#deepvac implemented based on PyTorch Framework
class Deepvac(object):
    def __init__(self, deepvac_config, is_forward_only=True):
        self.deepvac_config = deepvac_config
        self.core_config = deepvac_config.core
        self.initConfig()
        self.config.is_forward_only = is_forward_only
        self.config.branch = assertAndGetGitBranch(self.config.disable_git)
        self.init()

    def initConfig(self):
        if self.name() not in self.core_config.keys():
            self.core_config[self.name()] = AttrDict()
        self.config = self.core_config[self.name()]

    def addUserConfig(self, config_name, user_give=None, developer_give=None, is_user_mandatory=False):
        module_name = 'config.core.{}'.format(self.name())
        return addUserConfig(module_name, config_name, user_give=user_give, developer_give=developer_give, is_user_mandatory=is_user_mandatory)

    def name(self):
        return self.__class__.__name__

    def auditConfig(self):
        if not self.config.model_path and not self.config.jit_model_path:
            LOG.logE("both config.core.{}.model_path and config.core.{}.jit_model_path are not set, cannot do predict.".format(self.name(), self.name()), exit=True)
        #audit for ema
        if self.config.is_forward_only and self.config.ema:
            LOG.logE("Error: You must disable config.core.{}.ema in test only mode.".format(self.name()), exit=True)

    def _parametersInfo(self):
        param_info_list = [p.numel() for p in self.config.net.parameters() ]
        LOG.logI("config.net has {} parameters.".format(sum(param_info_list)))
        LOG.logI("config.net parameters detail: {}".format(['{}: {}'.format(name, p.numel()) for name, p in self.config.net.named_parameters()]))

    def initDevice(self):
        #to determine CUDA device, different in DDP
        self.config.device = torch.device(self.config.device)

    def initNetWithCode(self):
        if self.config.net is None:
            LOG.logE("You must implement and set config.core.{}.net to a torch.nn.Module instance in config.py.".format(self.name()), exit=True)
        if not isinstance(self.config.net, nn.Module):
            LOG.logE("You must set config.core.{}.net to a torch.nn.Module instance in config.py.".format(self.name()), exit=True)

    def auditStateDict(self, myconfig):
        state_dict = None
        if not myconfig.model_path:
            LOG.logI("config.core.{}.model_path not specified in config.py, network parametes will not be initialized from trained/pretrained model.".format(self.name()))
            return state_dict

        if myconfig.jit_model_path and self.config.is_forward_only:
            LOG.logI("config.core{}..jit_model_path set in forward-only mode in config.py, network parametes will be initialized from jit model rather than trained/pretrained model.".format(self.name()))
            return state_dict

        if myconfig.jit_model_path:
            LOG.logW("config.core{}.jit_model_path set in training mode in config.py, omit...".format(self.name()))

        LOG.logI('Loading State Dict from {}'.format(myconfig.model_path))
        state_dict = torch.load(myconfig.model_path, map_location=myconfig.device)
        #remove prefix begin
        prefix = 'module.'
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        if "state_dict" in state_dict.keys():
            state_dict = {f(key): value for key, value in state_dict['state_dict'].items()}
        else:
            state_dict = {f(key): value for key, value in state_dict.items()}
        #remove prefix end

        #audit on model file
        state_dict_keys = set(state_dict.keys())
        code_net_keys = set(myconfig.net.state_dict().keys())
        used_keys = code_net_keys & state_dict_keys
        unused_keys = state_dict_keys - code_net_keys
        missing_keys = code_net_keys - state_dict_keys
        LOG.logI('Missing keys before model_reinterpret_cast:{} | {}'.format(len(missing_keys), missing_keys))
        LOG.logI('Unused keys before model_reinterpret_cast:{} | {}'.format(len(unused_keys), unused_keys))
        LOG.logI('Used keys before model_reinterpret_cast:{}'.format(len(used_keys)))

        if myconfig.model_reinterpret_cast:
            LOG.logI("You enabled config.core.{}.model_reinterpret_cast in config.py, omit net parameter audit.".format(self.name()))
            return state_dict

        if len(used_keys) == 0:
            LOG.logE('Error: load NONE from pretrained model: {}'.format(myconfig.model_path), exit=True)

        if len(missing_keys) > 0:
            LOG.logW("There have missing network parameters, double check if you are using a mismatched trained model.")
            if not myconfig.network_audit_disabled:
                LOG.logE("If you know this risk, set config.core.{}.network_audit_disabled=True in config.py to omit this error.".format(self.name()), exit=True)
        return state_dict

    def initStateDict(self):
        self.config.state_dict = self.auditStateDict(self.config)

    def castStateDict(self, old_state_dict, myconfig):
        LOG.logI("config.core.{}.model_reinterpret_cast set to True in config.py, Try to reinterpret cast the model".format(self.name()))

        if self.config.model_path_omit_keys:
            LOG.logI("You have set config.core.{}.model_path_omit_keys: {}".format(self.name(), self.config.model_path_omit_keys))
            for k in self.config.model_path_omit_keys:
                LOG.logI("remove key {} from config.core.{}.model_path {}".format(k, self.name(), myconfig.model_path))
                old_state_dict.pop(k, None)

        state_dict = collections.OrderedDict()
        keys = list(old_state_dict.keys())
        model_path_keys_len = len(keys)
        if len(myconfig.net.state_dict() ) > model_path_keys_len:
            LOG.logW("config.core.{}.net has more parameters than config.core.{}.model_path({}), may has cast issues.".format(self.name(), self.name(), myconfig.model_path))

        real_idx = 0
        for _, name in enumerate(myconfig.net.state_dict()):
            if real_idx >= model_path_keys_len:
                LOG.logI("There alreay has no corresponding parameter in {} for {}".format(myconfig.model_path, name))
                continue

            if anyFieldsInConfig(name, self.config.net_omit_keys, self.config.net_omit_keys_strict):
                LOG.logI('found key to omit in config.core.{}.net: {}, continue...'.format(self.name(),name))
                continue
  
            if myconfig.net.state_dict()[name].size() == old_state_dict[keys[real_idx]].size():
                LOG.logI("cast pretrained model [{}] => config.core.{}.net [{}]".format(keys[real_idx], self.name(), name))
                state_dict[name] = old_state_dict[keys[real_idx]]
                real_idx += 1
                continue

            LOG.logE("cannot cast pretrained model [{}] => config.net [{}] due to parameter shape mismatch!".format(keys[real_idx], name))
            if myconfig.cast_state_dict_strict is False:
                real_idx += 1
                continue
            LOG.logE("If you know above risk, set config.core.{}.cast_state_dict_strict=False in config.py to omit this audit.".format(self.name()), exit=True)
            
        LOG.logI("Reinterpret cast the model succeeded.")
        return state_dict

    def loadStateDict(self):
        self.config.net = self.config.net.to(self.config.device)
        if not self.config.state_dict:
            LOG.logI("self.config.state_dict not initialized, omit loadStateDict()")
            return
        
        if self.config.model_reinterpret_cast:
            self.config.state_dict = self.castStateDict(self.config.state_dict, self.config)
            
        self.config.net.load_state_dict(self.config.state_dict, strict=False)

    def loadJitModel(self):
        if not self.config.jit_model_path:
            LOG.logI("config.core.{}.jit_model_path not specified, omit the loadJitModel".format(self.name()))
            return

        if not self.config.is_forward_only:
            LOG.logI("You are in training mode, omit the loadJitModel")
            return

        self.config.net = torch.jit.load(self.config.jit_model_path, map_location=self.config.device)

    def initTestLoader(self):
        #only init test_loader in base class
        if self.config.test_loader is None and self.config.is_forward_only:
            LOG.logE("You must set config.core.{}.test_loader in config.py".format(self.name()), exit=True)
        
        if self.config.test_loader is not None:
            LOG.logI("You set config.core.{}.test_loader to {} in config.py".format(self.name(), self.config.test_loader)) 

    def initEMA(self):
        if self.config.ema is not True:
            return

        LOG.logI("Notice: You have enabled ema, which will increase the memory usage.")
        self.config.ema_updates = 0
        self.config.ema_net = copy.deepcopy(self.config.net)
        self.config.ema_net.to(self.config.device)
        if self.config.ema_decay is None:
            self.config.ema_decay = lambda x: 0.9999 * (1 - math.exp(-x / 2000))

        for p in self.config.ema_net.parameters():
            p.requires_grad_(False) 

    def init(self):
        #init self.config.device
        self.initDevice()
        #init self.config.net
        self.initNetWithCode()
        #init EMA
        self.initEMA()
        #init self.model_dict
        self.initStateDict()
        #just load model after audit
        self.loadStateDict()
        #jit load model
        self.loadJitModel()
        #just print model parameters info
        self._parametersInfo()
        #init test_loader
        self.initTestLoader()

    def export3rd(self, output_file=None):
        export3rd(self.config, self.deepvac_config.cast, output_file)

    #For Deepvac user to reimplement
    def preIter(self):
        pass

    #for deepvac developer to reimplement
    def _preIter(self):
        if isinstance(self.config.data, (list, tuple)):
            data_len = len(self.config.data)
            if data_len == 2:
                self.config.sample, self.config.target = self.config.data
            elif data_len == 1:
                self.config.sample = self.config.data
            elif data_len > 2:
                self.config.sample = self.config.data[0]
                self.config.target = self.config.data[1:]
                LOG.logW("You should reimplement _preIter() in subclass {} since dataloader item lenth is {}.".format(self.name(), data_len))
            else:
                LOG.logE("You must reimplement _preIter() in subclass {} since dataloader item lenth is {}.".format(self.name(), data_len), exit=True)
        else:
            self.config.sample = self.config.data
        
        if not isinstance(self.config.sample, torch.Tensor):
            LOG.logE("You must reimplement _preIter() in subclass {} since got config.core.{}.sample type: {}.".format(self.name(), self.name(), type(self.config.sample)))
            LOG.logE("1st element in dataloader item must be torch.Tensor, i.e. your dataloader item should be [torch.Tensor, otherType,...].", exit=True)

        self.doFeedData2Device()

    def doFeedData2Device(self):
        self.config.sample = self.config.sample.to(self.config.device)
        if isinstance(self.config.target, torch.Tensor):
            self.config.target = self.config.target.to(self.config.device)

    #for deepvac developer to reimplement
    def _postIter(self):
        pass

    def postIter(self):
        pass

    def doForward(self):
        self.config.output = self.config.net(self.config.sample)

    def test(self):
        LOG.logI("config.core.{}.test_loader has been set, do test() with config.core.{}.test_loader".format(self.name(), self.name()))
        for self.config.test_step, self.config.data in enumerate(self.config.test_loader):
            self._preIter()
            self.preIter()
            self.doForward()
            LOG.logI('{}: [input shape: {}] [{}/{}]'.format(self.config.phase, self.config.sample.shape, self.config.test_step + 1, len(self.config.test_loader)))
            self.postIter()

    def testSample(self):
        self._preIter()
        self.preIter()
        self.doForward()
        LOG.logI('{}: [input shape: {}]'.format(self.config.phase, self.config.sample.shape))
        return self.config.output

    def doTest(self):
        if self.config.test_loader:
            return self.test()
        LOG.logE("You have to reimplement doTest() in subclass {} if you didn't set any valid input, e.g. config.core.{}.test_loader.".format(self.name(), self.name()), exit=True)

    def process(self, input_tensor):
        self.config.phase = 'TEST'
        self.config.net.eval()
        if input_tensor is not None:
            LOG.logI('You provided input_tensor at Deepvac(input_tensor), do net inference with this input_tensor.')
            self.config.sample = input_tensor
            return self.testSample()
        LOG.logI("You did not provide input_tensor at Deepvac(input_tensor)...")
        LOG.logI("doTest() is your last chance, you must have already reimplemented doTest() in subclass {}, right?".format(self.name()))
        x = self.doTest()
        if self.config.sample is None:
            LOG.logE("You must set self.config.sample in doTest() in your subclass {}, or cast model will fail.".format(self.name()), exit=True)
        return x

    def __call__(self, input_tensor=None):
        self.auditConfig()
        with torch.no_grad():
            x = self.process(input_tensor)
            #export 3rd
            self.export3rd()
        return x

#context manager for state switch
class deepvac_val_mode(object):
    def __init__(self, config):
        self.config = config
      
    def __enter__(self):
        self.config.is_train = False
        self.config.phase = 'VAL'
        self.config.dataset = self.config.val_dataset
        self.config.loader = self.config.val_loader
        self.config.batch_size = self.config.val_batch_size
        self.config.net.eval()
  
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.config.is_train = True
        self.config.phase = 'TRAIN'
        self.config.dataset = self.config.train_dataset
        self.config.loader = self.config.train_loader
        self.config.batch_size = self.config.train_batch_size
        self.config.net.train()

#base class for train pipeline    
class DeepvacTrain(Deepvac):
    def __init__(self, deepvac_config, is_forward_only=False):
        super(DeepvacTrain, self).__init__(deepvac_config, is_forward_only)
        self.initTrainContext()

    def auditConfig(self):
        #basic train config audit
        if self.config.train_dataset is None:
            LOG.logE("You must set config.core.{}.train_dataset in config.py".format(self.name()),exit=True)
        if self.config.val_dataset is None and not self.config.no_val:
            LOG.logE("You must set config.core.{}.val_dataset in config.py".format(self.name()),exit=True)

        #audit for amp
        if self.config.amp and self.config.device.type != 'cuda':
            LOG.logE("Error: amp can only be enabled when using cuda device", exit=True)

        #audit output dir
        if self.config.output_dir != 'output' and self.config.output_dir != './output':
            LOG.logW("According deepvac standard, you should set config.output_dir to [output] rather than [{}].".format(self.config.output_dir))

    def initTrainContext(self):
        self.config.epoch = 0
        self.config.iter = 0
        self.config.is_train = True
        self.config.phase = 'TRAIN'
        self.config.scaler = GradScaler()
        self.config.loader_time = AverageMeter()
        self.config.train_time = AverageMeter()
        self.initOutputDir()
        self.initSummaryWriter()
        self.initCriterion()
        self.initOptimizer()
        self.initScheduler()
        self.initCheckpoint()
        self.initTrainLoader()
        self.initValLoader()
        #must after initTrainLoader and initCheckpoint
        self.initEmaUpdates()

    def initOutputDir(self):
        self.config.output_dir = '{}/{}'.format(self.config.output_dir, self.config.branch)
        LOG.logI('model save dir: {}'.format(self.config.output_dir))
        #for DDP race condition
        os.makedirs(self.config.output_dir, exist_ok=True)

    def initSummaryWriter(self):
        event_dir = "{}/{}".format(self.config.log_dir, self.config.branch)
        self.config.writer = SummaryWriter(event_dir)
        if not self.config.tensorboard_port:
            return
        try:
            from tensorboard import program
        except Exception as e:
            LOG.logE(e.msg)
            LOG.logE("Make sure you have installed tensorboard with <pip install tensorboard>.", exit=True)
        tensorboard = program.TensorBoard()
        self.config.tensorboard_ip = '0.0.0.0' if self.config.tensorboard_ip is None else self.config.tensorboard_ip
        tensorboard.configure(argv=[None, '--host', str(self.config.tensorboard_ip),'--logdir', event_dir, "--port", str(self.config.tensorboard_port)])
        try:
            url = tensorboard.launch()
            LOG.logI('Tensorboard at {} '.format(url))
        except Exception as e:
            LOG.logE(e.msg, exit=True)

    def initCheckpoint(self):
        if not self.config.checkpoint_suffix or self.config.checkpoint_suffix == "":
            LOG.logI('Omit the checkpoint file since not specified...')
            return
        LOG.logI('Load checkpoint from {} folder'.format(self.config.output_dir))
        self.config.net.load_state_dict(torch.load(self.config.output_dir+'/model__{}'.format(self.config.checkpoint_suffix), map_location=self.config.device))
        state_dict = torch.load(self.config.output_dir+'/checkpoint__{}'.format(self.config.checkpoint_suffix), map_location=self.config.device)
        self.config.optimizer.load_state_dict(state_dict['optimizer'])
        self.config.epoch = state_dict['epoch']
        if self.config.scheduler:
            self.config.scheduler.load_state_dict(state_dict['scheduler'])
        
        if self.config.amp:
            self.initAmpScaler()

        if self.config.ema:
            self.config.ema_net.load_state_dict(state_dict['ema'])

    def initAmpScaler(self):
        LOG.logI("Will load scaler from checkpoint since you enabled amp, make sure the checkpoint was saved with amp enabled.")
        try:
            self.config.scaler.load_state_dict(state_dict["scaler"])
        except:
            LOG.logW("checkpoint was saved without amp enabled, so use fresh GradScaler instead.")
            self.config.scaler = GradScaler()

    def initCriterion(self):
        if self.config.criterion is None:
            LOG.logE("You should set config.core.{}.criterion in config.py, e.g. config.core.{}.criterion=torch.nn.CrossEntropyLoss()".format(self.name(), self.name()),exit=True)
        LOG.logI("You set config.core.{}.criterion to {}".format(self.name(), self.config.criterion))

    def initScheduler(self):
        if self.config.scheduler is None:
            LOG.logE("You must set config.core.{}.scheduler in config.py.".format(self.name()), exit=True)
        LOG.logI("You set config.core.{}.scheduler to {}".format(self.name(),self.config.scheduler))
        
    def initOptimizer(self):
        if self.config.optimizer is None:
            LOG.logE("You must set config.core.{}.optimizer in config.py.".format(self.name()),exit=True)
        LOG.logI("You set config.core.{}.optimizer to {} in config.py".format(self.name(),self.config.optimizer))

    def initTrainLoader(self):
        if self.config.train_loader is None:
            LOG.logE("You must set config.core.{}.train_loader in config.py, or reimplement initTrainLoader() API in your DeepvacTrain subclass {}.".format(self.name(),self.name()), exit=True)
        LOG.logI("You set config.core.{}.train_loader to {} in config.py".format(self.name(),self.config.train_loader))
        self.config.loader = self.config.train_loader

    def initValLoader(self):
        if self.config.no_val:
            LOG.logI("You specified config.core.{}.no_val={}, omit VAL phase.".format(self.name(),self.config.no_val))
            return
        if self.config.val_loader is None:
            LOG.logE("You must set config.core.{}.val_loader in config.py, or reimplement initValLoader() API in your DeepvacTrain subclass {}.".format(self.name(), self.name()), exit=True)
        LOG.logI("You set config.core.{}.val_loader to {} in config.py".format(self.name(),self.config.val_loader))

    def initEmaUpdates(self):
        if self.config.ema is not True:
            return
        self.config.ema_updates = self.config.epoch * len(self.config.train_loader) // self.config.nominal_batch_factor

    def addScalar(self, tag, value, step):
        self.config.writer.add_scalar(tag, value, step)

    def addImage(self, tag, image, step):
        self.config.writer.add_image(tag, image, step)

    @syszux_once
    def doGraph(self):
        if self.config.sample is None:
            LOG.logE("You must use doGraph in train() function.", exit=True)
        try:
            self.config.writer.add_graph(self.config.net, self.config.sample)
        except:
            LOG.logW("Tensorboard doGraph failed. You network foward may have more than one parameters?")
            LOG.logW("Seems you need to reimplement preIter function in subclass {}.".format(self.name()))

    def updateEMA(self):
        if self.config.ema is not True:
            return
        self.config.ema_updates += 1
        d = self.config.ema_decay(self.config.ema_updates)
        msd = self.config.net.state_dict()
        with torch.no_grad():
            for k, v in self.config.ema_net.state_dict().items():
                if not v.is_floating_point():
                    continue
                v *= d
                v += (1. - d) * msd[k].detach()

    #For Deepvac user to reimplement
    def preEpoch(self):
        pass

    #For Deepvac developer to reimplement
    def _preEpoch(self):
        pass

    #For Deepvac user to reimplement
    def postEpoch(self):
        pass

    #For Deepvac developer to reimplement
    def _postEpoch(self):
        pass

    def doLoss(self):
        self.config.loss = self.config.criterion(self.config.output, self.config.target)

    def doBackward(self):
        if self.config.amp:
            self.config.scaler.scale(self.config.loss).backward()
        else:
            self.config.loss.backward()

    def doOptimize(self):
        if self.config.iter % self.config.nominal_batch_factor != 0:
            return
        if self.config.amp:
            self.config.scaler.step(self.config.optimizer)
            self.config.scaler.update()
        else:
            self.config.optimizer.step()
        self.config.optimizer.zero_grad()

        if self.config.ema:
            self.updateEMA()

    def doSchedule(self):
        self.config.scheduler.step()

    def doValLog(self):
        if self.config.val_step % self.config.log_every != 0:
            return
        LOG.logI('{}: [{}][{}/{}]'.format(self.config.phase, self.config.epoch, self.config.val_step + 1, len(self.config.loader)))

    def doLog(self):
        if self.config.step % self.config.log_every != 0:
            return
        self.addScalar('{}/Loss'.format(self.config.phase), self.config.loss.item(), self.config.iter)
        self.addScalar('{}/LoadDataTime(secs/batch)'.format(self.config.phase), self.config.loader_time.val, self.config.iter)
        self.addScalar('{}/TrainTime(secs/batch)'.format(self.config.phase), self.config.train_time.val, self.config.iter)
        LOG.logI('{}: [{}][{}/{}] [Loss:{}  Lr:{}]'.format(self.config.phase, self.config.epoch, self.config.step + 1, len(self.config.loader),self.config.loss.item(),self.config.optimizer.param_groups[0]['lr']))

    def doSave(self):
        current_time = getPrintTime()
        #context for export 3rd
        LOG.logI("preparing save model with timefix: {}".format(current_time))
        with torch.no_grad(), deepvac_val_mode(self.config):
            file_partial_name = '{}__acc_{}__epoch_{}__step_{}__lr_{}'.format(current_time, self.config.acc, self.config.epoch, self.config.step, self.config.optimizer.param_groups[0]['lr'])
            state_file = '{}/model__{}.pth'.format(self.config.output_dir, file_partial_name)
            checkpoint_file = '{}/checkpoint__{}.pth'.format(self.config.output_dir, file_partial_name)

            net = self.config.ema_net if self.config.ema else self.config.net
            torch.save(net.state_dict(), state_file)
            #save checkpoint
            LOG.logI("saving model: {}".format(state_file))
            LOG.logI("saving checkpoint: {}".format(checkpoint_file))
            torch.save({
                'optimizer': self.config.optimizer.state_dict(),
                'epoch': self.config.epoch,
                'scheduler': self.config.scheduler.state_dict() if self.config.scheduler else None,
                'ema': self.config.ema_net.state_dict() if self.config.ema else None,
                'scaler': self.config.scaler.state_dict() if self.config.amp else None},  checkpoint_file)
            #tensorboard
            self.addScalar('{}/Accuracy'.format(self.config.phase), self.config.acc, self.config.iter)

            #export 3rd
            self.export3rd(file_partial_name)

    def initTickTock(self):
        self.config.train_time.reset()
        self.config.loader_time.reset()

    def initStepAndSaveNumber(self):
        loader_len = len(self.config.loader)
        LOG.logI("TRAIN DataLoader has length {}".format(loader_len))
        if loader_len == 0:
            LOG.logE("dataloader has length 0, make sure you have set correct DataLoader(and Dataset)", exit=True)
        save_every = loader_len//self.config.save_num
        save_list = list(range(0, loader_len + 1, save_every ))
        self.config.save_list = save_list[1:-1]
        LOG.logI('Model will be saved on step {} and the epoch end.'.format(self.config.save_list))
        self.addScalar('{}/LR'.format(self.config.phase), self.config.optimizer.param_groups[0]['lr'], self.config.epoch)

    def doTimekeeping(self):
        self.addScalar('{}/TrainTime(hours/epoch)'.format(self.config.phase), round(self.config.train_time.sum / 3600, 2), self.config.epoch)
        self.addScalar('{}/AverageBatchTrainTime(secs/epoch)'.format(self.config.phase), self.config.train_time.avg, self.config.epoch)
        self.addScalar('{}/AverageBatchLoadDataTime(secs/epoch)'.format(self.config.phase), self.config.loader_time.avg, self.config.epoch)

    def doIterTick(self):
        self.config.iter_tick = time.time()

    def doLoaderTock(self):
        self.config.loader_time.update(time.time() - self.config.iter_tick)

    def doTrainTock(self):
        self.config.train_time.update(time.time() - self.config.iter_tick)

    def doIterkeeping(self):
        self.config.iter += 1

    def doValAcc(self):
        self.config.acc = 0
        LOG.logW("You should reimplement doValAcc() to assign acc value to self.config.acc in your subclass {}.".format(self.name()))

    def train(self):
        self.config.net.train()
        self.initStepAndSaveNumber()
        self.initTickTock()
        #for multi loader user case
        self.config.loader = self.config.train_loader
        self.preEpoch()
        self._preEpoch()
        self.doIterTick()
        for self.config.step, self.config.data in enumerate(self.config.loader):
            self.doLoaderTock()
            self._preIter()
            self.preIter()
            self.doGraph()
            with autocast(enabled=self.config.amp if self.config.amp else False):
                self.doForward()
                self.doLoss()
            self.doBackward()
            self.doOptimize()
            self.doLog()
            self.postIter()
            self.doIterkeeping()
            self.doTrainTock()
            if self.config.step in self.config.save_list:
                self.val()
                self.doSave()
            self.doIterTick()
        #epoch end
        self.postEpoch()
        #must before schedule
        self.doSave()
        self.doSchedule()
        self.doTimekeeping()

    def val(self, smoke=False):
        if self.config.no_val:
            return
        with torch.no_grad(), deepvac_val_mode(self.config):
            LOG.logI('Phase {} started...'.format(self.config.phase))
            self.preEpoch()
            for self.config.val_step, self.config.data in enumerate(self.config.loader):
                self._preIter()
                self.preIter()
                self.doForward()
                self.doLoss()
                self.postIter()
                if smoke:
                    self.export3rd()
                    return
                self.doValLog()
            self.postEpoch()
            self.doValAcc()
            LOG.logI('Phase {} end...'.format(self.config.phase))

    def processAccept(self):
        pass

    def process(self):
        self.config.optimizer.zero_grad()
        epoch_start = self.config.epoch
        for self.config.epoch in range(epoch_start, self.config.epoch_num):
            LOG.logI('Epoch {} started...'.format(self.config.epoch))
            self.train()
            self.val()
            self.processAccept()

    def __call__(self):
        self.auditConfig()
        self.val(smoke=True)
        self.process()

class DeepvacDDP(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(DeepvacDDP,self).__init__(deepvac_config)
        assert self.config.train_sampler is not None, "You should define config.core.{}.train_sampler in config.py when training with DDP mode.".format(self.name())

    def initDevice(self):
        super(DeepvacDDP, self).initDevice()
        parser = argparse.ArgumentParser(description='DeepvacDDP')
        parser.add_argument('vars', nargs='*')
        parser.add_argument("--gpu", default=-1, type=int, help="gpu")
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        self.config.args = parser.parse_args()
        self.config.map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config.args.rank}
        #in DDP, device may come from command line
        if self.config.args.gpu:
            self.config.device = torch.device(self.config.args.gpu)
        torch.cuda.set_device(self.config.args.gpu)

    def initDDP(self):
        LOG.logI("Start dist.init_process_group {} {}@{} on {}".format(self.config.dist_url, self.config.args.rank, self.config.world_size - 1, self.config.args.gpu))
        dist.init_process_group(backend='nccl', init_method=self.config.dist_url, world_size=self.config.world_size, rank=self.config.args.rank)
        self.config.net = torch.nn.parallel.DistributedDataParallel(self.config.net, device_ids=[self.config.args.gpu])
        LOG.logI("Finish dist.init_process_group {} {}@{} on {}".format(self.config.dist_url, self.config.args.rank, self.config.world_size - 1, self.config.args.gpu))
        self.config.train_sampler = torch.utils.data.distributed.DistributedSampler(self.config.train_dataset)
        if self.config.train_loader is not None:
            LOG.logW("Warning: config.train_loader will be override by Deepvac DDP is enabled. reimplement initDDP() API in your DeepvacTrain subclass {} to change this behaviour.".format(self.name()))
        self.config.train_loader = DataLoader(
            self.config.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            sampler=self.config.train_sampler,
            collate_fn=self.config.collate_fn
        )

    def initTrainContext(self):
        self.initDDP()
        super(DeepvacDDP,self).initTrainContext()

    def initSummaryWriter(self):
        if self.config.args.rank != 0:
            return
        super(DeepvacDDP, self).initSummaryWriter()

    def _preEpoch(self):
        self.config.train_sampler.set_epoch(self.config.epoch)

    def doSave(self):
        if self.config.args.rank != 0:
            return
        super(DeepvacDDP, self).doSave()

    def addScalar(self, tag, value, step):
        if self.config.args.rank != 0:
            return
        super(DeepvacDDP, self).addScalar(tag, value, step)

    def addImage(self, tag, image, step):
        if self.config.args.rank != 0:
            return
        super(DeepvacDDP, self).addImage(tag, image, step)

    @syszux_once
    def doGraph(self):
        if self.config.args.rank != 0:
            return
        super(DeepvacDDP, self).doGraph()

if __name__ == "__main__":
    from config import config as deepvac_config
    vac = DeepvacTrain(deepvac_config)
    vac()
