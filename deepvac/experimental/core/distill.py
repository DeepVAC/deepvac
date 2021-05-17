import torch.nn as nn
from ...core import DeepvacTrain
from ...utils import LOG

class DeepvacDistill(DeepvacTrain):
    def auditConfig(self):
        super(DeepvacDistill, self).auditConfig()
        if self.config.teacher.scheduler is None:
            LOG.logE("You must set config.train.teacher.scheduler in config.py.", exit=True)
        LOG.logI("You set config.train.teacher.scheduler to {}".format(self.config.teacher.scheduler))

        if self.config.teacher.optimizer is None:
            LOG.logE("You must set config.train.teacher.optimizer in config.py.",exit=True)
        LOG.logI("You set config.train.teacher.optimizer to {} in config.py".format(self.config.teacher.optimizer))

    def initNetWithCode(self):
        super(DeepvacDistill, self).initNetWithCode()
        if self.config.teacher.net is None:
            LOG.logE("You must implement and set config.train.teacher.net to a torch.nn.Module instance in config.py.", exit=True)
        if not isinstance(self.config.teacher.net, nn.Module):
            LOG.logE("You must set config.train.teacher.net to a torch.nn.Module instance in config.py.", exit=True)

    def initStateDict(self):
        super(DeepvacDistill, self).initStateDict()
        LOG.logI('Loading State Dict from {}'.format(self.config.teacher.model_path))
        self.config.teacher.state_dict = self.auditStateDict(self.config.teacher)
        
    def loadStateDict(self):
        super(DeepvacDistill, self).loadStateDict()
        self.config.teacher.net = self.config.teacher.net.to(self.config.device)
        if not self.config.teacher.state_dict:
            LOG.logI("self.config.teacher.state_dict not initialized, omit loadStateDict()")
            return
        
        if self.config.model_reinterpret_cast:
            self.config.teacher.state_dict = self.castStateDict(self.config.teacher)
            
        self.config.teacher.net.load_state_dict(self.config.teacher.state_dict, strict=False)

    def doForward(self):
        super(DeepvacDistill, self).doForward()
        self.config.teacher.output = self.config.teacher.net(self.config.sample)

    def initCriterion(self):
        super(DeepvacDistill, self).initCriterion()
        if self.config.teacher.criterion is None:
            LOG.logE("You must set config.train.criterion in config.py, e.g. config.train.teacher.criterion=torch.nn.CrossEntropyLoss()",exit=True)
        LOG.logI("You set config.train.teacher.criterion to {}".format(self.config.teacher.criterion))

    def doLoss(self):
        super(DeepvacDistill, self).doLoss()
        LOG.logE("You have to reimplement doLoss() in DeepvacDistill subclass and set config.train.teacher.loss", exit=True)

    def doBackward(self):
        super(DeepvacDistill, self).doBackward()
        if self.config.teacher.use_backword is True:
            self.config.teacher.loss.backward()
