import os
import torch
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
        #init self.net
        self.initNet()

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

    def initDevice(self):
        #to determine CUDA device
        self.device = torch.device("cuda")

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

    def process(self):
        raise Exception("Not implemented!")

    def report(self):
        pass

    def __call__(self,input):
        self.setInput(input)
        self.process()
        return self.getOutput()
        

if __name__ == "__main__":
    from conf import config as deepvac_config
    vac = DeepVAC(deepvac_config)
    vac()
