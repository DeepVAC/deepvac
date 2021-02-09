import copy
import os
import sys
from datetime import datetime
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.quantization.fuser_method_mappings import DEFAULT_OP_LIST_TO_FUSER_METHOD
from torch.quantization import quantize_dynamic_jit, per_channel_dynamic_qconfig
from torch.quantization import get_default_qconfig, quantize_jit
import time
import subprocess
import tempfile
from enum import Enum
from typing import Any, Callable
from .syszux_annotation import *
from .syszux_log import LOG,getCurrentGitBranch
from .syszux_helper import AverageMeter
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    LOG.logE("Deepvac has dependency on tensorboard, please install tensorboard first, e.g. [pip3 install tensorboard]", exit=True)

from torch.quantization import QuantStub, DeQuantStub

def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def get_fuser_module_index(mod_list):
    rc = []
    if len(mod_list) < 2:
        return rc
    keys = sorted(list(DEFAULT_OP_LIST_TO_FUSER_METHOD.keys()), key=lambda x: len(x), reverse=True)
    mod2fused_list = [list(x) for x in keys]

    for mod2fused in mod2fused_list:
        if len(mod2fused) > len(mod_list):
            continue
        mod2fused_idx = [(i, i+len(mod2fused)) for i in range(len(mod_list) - len(mod2fused) + 1) if mod_list[i:i+len(mod2fused)] == mod2fused]
        if not mod2fused_idx:
            continue

        for idx in mod2fused_idx:
            start,end = idx
            mod_list[start: end] = [None] * len(mod2fused)

        rc.extend(mod2fused_idx)

    return rc

def auto_fuse_model(model):
    module_names = []
    module_types = []
    for name, m in model.named_modules():
        module_names.append(name)
        module_types.append(type(m))

    if len(module_types) < 2:
        return model

    module_idxs = get_fuser_module_index(module_types)
    modules_to_fuse = [module_names[mi[0]:mi[1]] for mi in module_idxs]
    new_model = torch.quantization.fuse_modules(model, modules_to_fuse)
    return new_model

class DeQuantStub(nn.Module):
    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x: Any) -> Any:
        return x

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for sample, target in data_loader:
            model(sample)

class DeepvacQAT(torch.nn.Module):
    def __init__(self, net2qat):
        super(DeepvacQAT, self).__init__()
        self.quant = QuantStub()
        self.net2qat = auto_fuse_model(net2qat)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.net2qat(x)
        x = self.dequant(x)
        return x

class SaveModel(object):
    def __init__(self, input_net, output_file, backend = 'fbgemm'):
        self.input_net = copy.deepcopy(input_net)
        self.input_net.cpu().eval()
        self.output_file = output_file
        self.dq_output_file = '{}.dq'.format(output_file)
        self.sq_output_file = '{}.sq'.format(output_file)
        self.d_qconfig_dict = {'': per_channel_dynamic_qconfig}
        self.s_qconfig_dict = {'': get_default_qconfig(backend) }
        self.ts = None

    def getConvertedNetFromQAT(self, net):
        toq_net = copy.deepcopy(net)
        toq_net.eval()
        torch.quantization.convert(toq_net.cpu(), inplace=True)
        return toq_net

    def export(self, input_sample=None):
        if isinstance(input_sample, torch.Tensor):
            input_sample = input_sample.cpu()

        with torch.no_grad():
            self._export(input_sample)

    def saveByInputOrNot(self, input_sample=None):
        if self.ts is None:
            self.export(input_sample)
        
        freeze_ts = torch.jit.freeze(self.ts)
        torch.jit.save(freeze_ts, self.output_file)

    def saveDQ(self, input_sample=None):
        if self.ts is None:
            self.export(input_sample)
        LOG.logI("Pytorch model dynamic quantize starting, will save model in {}".format(self.dq_output_file))
        quantized_model = quantize_dynamic_jit(self.ts, self.d_qconfig_dict)
        torch.jit.save(quantized_model, self.dq_output_file)
        LOG.logI("Pytorch model dynamic quantize succeeded, saved model in {}".format(self.dq_output_file))

    def saveSQ(self, loader, input_sample=None):
        if loader is None:
            LOG.logE("You enabled config.static_quantize_dir, but didn't provide self.test_loader in forward-only mode, or self.val_loader in train mode.", exit=True)
        if self.ts is None:
            self.export(input_sample)

        LOG.logI("Pytorch model static quantize starting, will save model in {}".format(self.sq_output_file))
        quantized_model = quantize_jit(self.ts, self.s_qconfig_dict, calibrate, [loader], inplace=False,debug=False)
        torch.jit.save(quantized_model, self.sq_output_file)
        LOG.logI("Pytorch model static quantize succeeded, saved model in {}".format(self.sq_output_file))

class SaveModelByTrace(SaveModel):
    def _export(self, input_sample):
        LOG.logI("SaveModelByTrace: {} ...".format(self.output_file))
        self.ts = torch.jit.trace(self.input_net, input_sample).eval()

class SaveModelByScript(SaveModel):
    def _export(self, input_sample=None):
        LOG.logI("SaveModelByScript: {} ...".format(self.output_file))
        self.ts = torch.jit.script(self.input_net).eval()

class SaveModelByQAT(SaveModelByScript):
    def _export(self, input_sample=None):
        LOG.logI("SaveModelByQAT: {} ...".format(self.output_file))
        qat_net = self.getConvertedNetFromQAT(self.input_net)
        self.ts = torch.jit.script(qat_net).eval()
    
#deepvac implemented based on PyTorch Framework
class Deepvac(object):
    def __init__(self, deepvac_config):
        self._mandatory_member = dict()
        self._mandatory_member_name = ['']
        self.input_output = {'input':[], 'output':[]}
        self.use_original_net_pre_qat = False
        self.conf = deepvac_config
        if self.conf.is_forward_only is None:
            self.conf.is_forward_only = True
        self.assertInGit()
        #init self.net
        self.initNet()
        self.initNetPost()

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

        if self.branch in ['master','main']:
            return

        LOG.logE('According to deepvac standard, git branch name should be master or main, or start from LTS_ or PROTO_: {}'.format(self.branch), exit=True)

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
        #audit for arch
        for name in self._mandatory_member_name:
            if name not in self._mandatory_member:
                LOG.logE("Error! self.{} must be definded in your subclass.".format(name),exit=True)

        #audit for quantize
        l = [self.conf.static_quantize_dir, self.conf.dynamic_quantize_dir,self.conf.qat_dir]
        l2 = [x for x in l if x]
        if len(l2) > 1:
            LOG.logE("Error: [static_quantize_dir, dynamic_quantize_dir, qat_dir] are exclusive for each other. You can only enable one of them in a train task.", exit=True)

        if self.conf.dynamic_quantize_dir and not any([self.conf.script_model_dir, self.conf.trace_model_dir]):
            LOG.logE("Error: to enable config.dynamic_quantize_dir, you must enable config.script_model_dir or config.trace_model_dir first.", exit=True)
        
        if self.conf.static_quantize_dir and not any([self.conf.script_model_dir, self.conf.trace_model_dir]):
            LOG.logE("Error: to enable config.static_quantize_dir, you must enable config.script_model_dir or config.trace_model_dir first.", exit=True)

        #audit for amp
        if self.conf.amp and self.device.type != 'cuda':
            LOG.logE("Error: amp can only be enabled when using cuda device", exit=True)

        if self.conf.qat_dir and self.conf.trace_model_dir:
            LOG.logE("Error: [qat_dir and trace_model_dir] are exclusive for each other. You can only enable one of them in a train task.", exit=True)

        #audit datalodaer
        if self.train_loader is None:
            LOG.logE("Error: self.train_loader not initialized. Have you reimplemented initTrainLoader() API?", exit=True)
        
        if self.val_loader is None:
            LOG.logE("Error: self.val_loader not initialized. Have you reimplemented initValLoader() API?", exit=True)

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

    def _parametersInfo(self):
        param_info_list = [p.numel() for p in self.net.parameters() ]
        LOG.logI("self.net has {} parameters.".format(sum(param_info_list)))
        LOG.logI("self.net parameters detail: {}".format(['{}: {}'.format(name, p.numel()) for name, p in self.net.named_parameters()]))

    def initNet(self):
        self.initLog()
        #init self.device
        self.initDevice()
        #init self.net
        self.initNetWithCode()
        #init quantize stuff
        self.initNetWithQuantize()
        self.initEMA()
        #init self.model_dict
        self.initStateDict()
        #just load model after audit
        self.loadStateDict()
        #jit load model
        self.loadJitModel()
        #just print model parameters info
        self._parametersInfo()
        self.initTestLoader()

    def initNetPost(self):
        self.xb = torch.Tensor().to(self.device)
        self.sample = None

    def initDevice(self):
        #to determine CUDA device, different in DDP
        self.device = torch.device(self.conf.device)

    def initNetWithQuantize(self):
        self.qat_net_prepared = None
        if self.conf.qat_dir:
            self.prepareQAT()

    def initTestLoader(self):
        self.test_loader = None
        LOG.logW("You must reimplement initTestLoader() to initialize self.test_loader")

    def initEMA(self):
        self.ema = None
        if self.conf.ema is None:
            return
        
        LOG.logI("Notice: You have enabled ema, which will increase the memory usage.")
        self.ema_updates = 0
        self.ema = copy.deepcopy(self.net)
        self.ema.to(self.device)
        if self.conf.ema_decay is None:
            self.conf.ema_decay = lambda x: 0.9999 * (1 - math.exp(-x / 2000))

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def updateEMA(self):
        if self.conf.ema is None:
            return
        self.ema_updates += 1
        d = self.conf.ema_decay(self.ema_updates)
        msd = self.net.state_dict()
        with torch.no_grad():
            for k, v in self.ema.state_dict().items():
                if not v.is_floating_point():
                    continue
                v *= d
                v += (1. - d) * msd[k].detach()

    def initNetWithCode(self):
        self.net = None
        LOG.logE("You must reimplement initNetWithCode() to initialize self.net", exit=True)

    def initStateDict(self):
        self.state_dict = None
        if not self.conf.model_path:
            LOG.logI("config.model_path not specified, omit the initStateDict")
            return

        if self.conf.jit_model_path and self.conf.is_forward_only:
            LOG.logI("config.jit_model_path specified in forward-only mode, omit the initStateDict")
            return

        LOG.logI('Loading State Dict from {}'.format(self.conf.model_path))
        self.state_dict = torch.load(self.conf.model_path, map_location=self.device)
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
        LOG.logI('Missing keys:{} | {}'.format(len(missing_keys), missing_keys))
        LOG.logI('Unused keys:{} | {}'.format(len(unused_keys), unused_keys))
        LOG.logI('Used keys:{}'.format(len(used_keys)))

        origin_used_keys_num = 0
        origin_missing_keys_num = 99999

        if self.conf.qat_dir:
            origin_code_net_keys = set(self.net.net2qat.state_dict().keys())
            origin_used_keys_num = len(origin_code_net_keys & state_dict_keys)
            origin_unused_keys_num = len(state_dict_keys - origin_code_net_keys)
            origin_missing_keys_num = len(origin_code_net_keys - state_dict_keys)
            LOG.logI('Origin missing keys:{}'.format(origin_missing_keys_num))
            LOG.logI('Origin unused keys:{}'.format(origin_unused_keys_num))
            LOG.logI('Origin used keys:{}'.format(origin_used_keys_num))

        if len(used_keys) == 0 and origin_used_keys_num == 0:
            LOG.logE('Error: load NONE from pretrained model', exit=True)

        if len(missing_keys) > 0 and origin_missing_keys_num > 0:
            LOG.logW("There have missing network parameters, double check if you are using a mismatched trained model.")

        if self.conf.qat_dir and origin_used_keys_num > len(used_keys):
            self.use_original_net_pre_qat = True

    def loadStateDict(self):
        self.net = self.net.to(self.device)
        if not self.state_dict:
            LOG.logI("self.state_dict not initialized, omit loadStateDict()")
            return
        
        if self.conf.qat_dir and self.use_original_net_pre_qat:
            self.net.net2qat.load_state_dict(self.state_dict, strict=False)
        else:
            self.net.load_state_dict(self.state_dict, strict=False)
        self.net.eval()
    
    def loadJitModel(self):
        if not self.conf.jit_model_path:
            LOG.logI("config.jit_model_path not specified, omit the loadJitModel")
            return

        if not self.conf.is_forward_only:
            LOG.logI("You are in training mode, omit the loadJitModel")
            return

        exclusive_options = [self.conf.trace_model_dir, self.conf.script_model_dir, self.conf.static_quantize_dir, self.conf.dynamic_quantize_dir]
        if any(exclusive_options):
            LOG.logE("config.jit_model_path is exclusive with {} in TEST MODE.".format(exclusive_options),exit=True)

        self.net = torch.jit.load(self.conf.jit_model_path, map_location=self.device)
        self.net.eval()
        self.net = self.net.to(self.device)

    def initLog(self):
        pass

    def process(self):
        LOG.logE("You must reimplement process() to process self.input_output['input']", exit=True)

    def processSingle(self,sample):
        self.sample = sample
        return self.net(self.sample)

    def __call__(self, input=None):
        if not self.state_dict and not self.conf.jit_model_path:
            LOG.logE("self.state_dict not initialized, cannot do predict.", exit=True)

        if input is not None:
            self.setInput(input)
        
        self.smokeTestForExport3rd(input)

        with torch.no_grad():
            self.process()

        return self.getOutput()

    def exportTorchViaTrace(self, sample=None, output_trace_file=None):
        if not self.conf.trace_model_dir:
            return
        if sample is None and self.sample is None:
            LOG.logE("either call exportTorchViaTrace and pass value to pamameter sample, or call exportTorchViaTrace in Train mode.", exit=True)

        if sample is not None:
            self.sample = sample
        
        if self.sample is None:
            LOG.logE("You enabled config.trace_model_dir, but didn't provide input. Please add input_tensor first, e.g. x = Deepvac(input_tensor)",exit=True)

        if output_trace_file is None:
            output_trace_file = self.conf.trace_model_dir

        LOG.logI("config.trace_model_dir found, save trace model to {}...".format(output_trace_file))

        net = self.ema if self.conf.ema else self.net
        save_model = SaveModelByTrace(net, output_trace_file)
        save_model.saveByInputOrNot(self.sample)

        if self.conf.dynamic_quantize_dir:
            LOG.logI("You have enabled config.dynamic_quantize_dir, will dynamic quantize the model...")
            save_model.saveDQ(self.sample)
        
        if self.conf.static_quantize_dir:
            LOG.logI("You have enabled config.static_quantize_dir, will static quantize the model...")
            loader = self.test_loader if self.conf.is_forward_only else self.val_loader
            save_model.saveSQ(loader, self.sample)

    def exportTorchViaScript(self, output_script_file=None):
        if not self.conf.script_model_dir:
            return

        if output_script_file is None:
            output_script_file = self.conf.script_model_dir

        LOG.logI("config.script_model_dir found, save script model to {}...".format(output_script_file))

        net = self.ema if self.conf.ema else self.net
        save_model = SaveModelByQAT(net, "{}.qat".format(output_script_file)) if self.conf.qat_dir else SaveModelByScript(net, output_script_file) 
        save_model.saveByInputOrNot()

        if self.conf.qat_dir:
            LOG.logI("You have enabled config.qat_dir, omit other type quantize and return...")
            return

        if self.conf.dynamic_quantize_dir:
            LOG.logI("You have enabled config.dynamic_quantize_dir, will dynamic quantize the model...")
            save_model.saveDQ()

        if self.conf.static_quantize_dir:
            LOG.logI("You have enabled config.static_quantize_dir, will static quantize the model...")
            loader = self.test_loader if self.conf.is_forward_only else self.val_loader
            save_model.saveSQ(loader)

    def exportNCNN(self, output_ncnn_file=None):
        if not self.conf.ncnn_model_dir:
            return

        if not self.conf.onnx2ncnn:
            LOG.logE("You must set the onnx2ncnn executable program path in config file. If you want to compile onnx2ncnn tools, reference https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86 ", exit=True)

        if output_ncnn_file is None:
            output_ncnn_file = self.conf.ncnn_model_dir

        self.ncnn_arch_dir = '{}.param'.format(output_ncnn_file)
        try:
            import onnx
            from onnxsim import simplify
        except:
            LOG.logE("You must install onnx and onnxsim package if you want to convert pytorch to ncnn.")

        if not self.conf.onnx_model_dir:
            f = tempfile.NamedTemporaryFile()
            self.conf.onnx_model_dir = f.name

        self.exportONNX()

        cmd = self.conf.onnx2ncnn + " " + self.conf.onnx_model_dir + " " + self.conf.ncnn_arch_dir + " " + output_ncnn_file
        pd = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pd.stderr.read() != b"":
            LOG.logE(pd.stderr.read() + b". Error occured when export ncnn model. We try to simplify the model first")
            model_op, check_ok = simplify(self.conf.onnx_model_dir, check_n=3, perform_optimization=True, skip_fuse_bn=True,  skip_shape_inference=False)
            onnx.save(model_op, self.conf.onnx_model_dir)
            if not check_ok:
                LOG.logE("Maybe something wrong when simplify the model, we can't guarantee generate model is right")
            else:
                LOG.logI("Simplify model succeed")
            subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if pd.stderr.read() != b"":
                LOG.logE(pd.stderr.read() + b". we can't guarantee generate model is right")

        LOG.logI("Pytorch model convert to NCNN model succeed, save ncnn param file in {}, save ncnn bin file in {}".format(self.conf.ncnn_arch_dir, output_ncnn_file))

    def exportCoreML(self, output_coreml_file=None):
        if not self.conf.coreml_model_dir:
            return

        if output_coreml_file is None:
            output_coreml_file = self.conf.coreml_model_dir

        try:
            from onnx_coreml import convert
        except:
            LOG.logE("You must install onnx-coreml, coremltools package if you want to convert PyTorch to CoreML model. E.g. pip install --upgrade onnx-coreml coremltools")

        input_names = ["deepvac_in"]
        output_names = ["deepvac_out"]

        if not self.conf.onnx_model_dir:
            f = tempfile.NamedTemporaryFile()
            self.conf.onnx_model_dir = f.name

        #to onnx
        net = self.ema if self.conf.ema else self.net
        torch.onnx.export(net, self.sample, self.conf.onnx_model_dir, verbose=True, input_names=input_names, output_names=output_names)
        #onnx2coreml
        model_coreml = convert(model=self.conf.onnx_model_dir, preprocessing_args= self.conf.coreml_preprocessing_args, mode=self.conf.coreml_mode, image_input_names=['deepvac_in'],
            class_labels=self.conf.coreml_class_labels, predicted_feature_name='deepvac_out', minimum_ios_deployment_target=self.conf.minimum_ios_deployment_target)

        # Save the CoreML model
        coreml_model.save(output_coreml_file)

    def exportONNX(self, output_onnx_file=None):
        if not self.conf.onnx_model_dir:
            return
        if output_onnx_file is None:
            output_onnx_file = self.onnx_model_dir
        else:
            self.onnx_model_dir = output_onnx_file

        net = self.ema if self.conf.ema else self.net
        torch.onnx._export(net, self.sample, output_onnx_file, export_params=True)
        LOG.logI("Pytorch model convert to ONNX model succeed, save model in {}".format(output_onnx_file))

    @syszux_once
    def smokeTestForExport3rd(self, input=None):
        #exportNCNN must before exportONNX !!!
        self.exportONNX()
        self.exportNCNN()
        self.exportCoreML()
        self.exportTorchViaTrace(input)
        self.exportTorchViaScript()

    def prepareQAT(self):
        if not self.conf.qat_dir:
            return

        if self.conf.is_forward_only:
            LOG.logI("You are in forward_only mode, omit the parepareQAT()")
            return

        LOG.logI("You have enabled QAT, this step is only for prepare.")

        if self.qat_net_prepared:
            LOG.logE("Error: You have already prepared the model for QAT.", exit=True)

        backend = 'fbgemm'
        if self.conf.quantize_backend:
            backend = self.conf.quantize_backend
        self.qat_net_prepared = DeepvacQAT(self.net).to(self.device)
        self.qat_net_prepared.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(self.qat_net_prepared, inplace=True)
        #after this, train net will be transfered to QAT !
        self.net = self.qat_net_prepared

    def loadDB(self, db_path):
        self.xb = torch.load(db_path).to(self.device)

    def addEmb2DB(self, emb):
        self.xb = torch.cat((self.xb, emb))

    def saveDB(self, db_path):
        torch.save(self.xb, db_path)

    def search(self, xq, k=1):
        D = []
        I = []
        if k < 1 or k > 10:
            LOG.logE('illegal nearest neighbors parameter k(1 ~ 10): {}'.format(k))
            return D, I

        distance = torch.norm(self.xb - xq, dim=1)
        for i in range(k):
            values, indices = distance.kthvalue(i+1)
            D.append(values.item())
            I.append(indices.item())
        return D, I

class DeepvacTrain(Deepvac):
    def __init__(self, deepvac_config):
        deepvac_config.is_forward_only=False
        super(DeepvacTrain,self).__init__(deepvac_config)
        self.initTrainParameters()
        self.initTrainContext()

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

    def initTrainContext(self):
        self.scheduler = None
        self.initOutputDir()
        self.initSummaryWriter()
        self.initCriterion()
        self.initOptimizer()
        self.initScheduler()
        self.initCheckpoint()
        self.initTrainLoader()
        self.initValLoader()

    def initTrainParameters(self):
        self.dataset = None
        self.loader = None
        self.target = None
        self.epoch = 0
        self.step = 0
        self.iter = 0
        # Creates a GradScaler once at the beginning of training.
        self.scaler = GradScaler()
        self.train_time = AverageMeter()
        self.load_data_time = AverageMeter()
        self.data_cpu2gpu_time = AverageMeter()
        self._mandatory_member_name = ['train_dataset','val_dataset','train_loader','val_loader','net','criterion','optimizer']

    def initOutputDir(self):
        if self.conf.output_dir != 'output' and self.conf.output_dir != './output':
            LOG.logW("According deepvac standard, you should set config.output_dir to [output] rather than [{}].".format(self.conf.output_dir))

        self.output_dir = '{}/{}'.format(self.conf.output_dir, self.branch)
        LOG.logI('model save dir: {}'.format(self.output_dir))
        #for DDP race condition
        os.makedirs(self.output_dir, exist_ok=True)

    def initSummaryWriter(self):
        event_dir = "{}/{}".format(self.conf.log_dir, self.branch)
        self.writer = SummaryWriter(event_dir)
        if not self.conf.tensorboard_port:
            return
        from tensorboard import program
        tensorboard = program.TensorBoard()
        self.conf.tensorboard_ip = '0.0.0.0' if self.conf.tensorboard_ip is None else self.conf.tensorboard_ip
        tensorboard.configure(argv=[None, '--host', str(self.conf.tensorboard_ip),'--logdir', event_dir, "--port", str(self.conf.tensorboard_port)])
        try:
            url = tensorboard.launch()
            LOG.logI('Tensorboard at {} '.format(url))
        except Exception as e:
            LOG.logE(e.msg)

    def initCriterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()
        LOG.logW("You should reimplement initCriterion() to initialize self.criterion, unless CrossEntropyLoss() is exactly what you need")

    def initCheckpoint(self):
        if not self.conf.checkpoint_suffix or self.conf.checkpoint_suffix == "":
            LOG.logI('Omit the checkpoint file since not specified...')
            return
        LOG.logI('Load checkpoint from {} folder'.format(self.output_dir))
        self.net.load_state_dict(torch.load(self.output_dir+'/model__{}'.format(self.conf.checkpoint_suffix), map_location=self.device))
        state_dict = torch.load(self.output_dir+'/checkpoint__{}'.format(self.conf.checkpoint_suffix), map_location=self.device)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        if self.conf.amp:
            LOG.logI("Will load scaler from checkpoint since you enabled amp, make sure the checkpoint was saved with amp enabled.")
            try:
                self.scaler.load_state_dict(state_dict["scaler"])
            except:
                LOG.logI("checkpoint was saved without amp enabled, so use fresh GradScaler instead.")
                self.scaler = GradScaler()

        self.epoch = state_dict['epoch']
        if self.conf.ema:
            self.ema.load_state_dict(state_dict['ema'])

    def initScheduler(self):
        if isinstance(self.conf.lr_step, list):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.conf.lr_step,self.conf.lr_factor)
        elif isinstance(self.conf.lr_step, Callable):
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.conf.lr_step)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.conf.lr_step,self.conf.lr_factor)
        LOG.logW("You should reimplement initScheduler() to initialize self.scheduler, unless lr_scheduler.StepLR() or lr_scheduler.MultiStepLR() is exactly what you need")

    def initTrainLoader(self):
        self.train_loader = None
        LOG.logE("You must reimplement initTrainLoader() to initialize self.train_loader", exit=True)

    def initValLoader(self):
        self.val_loader = None
        LOG.logE("You must reimplement initTrainLoader() to initialize self.val_loader", exit=True)

    def initOptimizer(self):
        self.initSgdOptimizer()
        LOG.logW("You should reimplement initOptimizer() to initialize self.optimizer, unless SGD is exactly what you need")

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

    def addScalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def addImage(self, tag, image, step):
        self.writer.add_image(tag, image, step)

    @syszux_once
    def addGraph(self, input):
        self.writer.add_graph(self.net, input)

    def earlyIter(self):
        start = time.time()
        self.sample = self.sample.to(self.device)
        self.target = self.target.to(self.device)
        if not self.is_train:
            return
        self.data_cpu2gpu_time.update(time.time() - start)
        try:
            self.addGraph(self.sample)
        except:
            LOG.logW("Tensorboard addGraph failed. You network foward may have more than one parameters?")
            LOG.logW("Seems you need reimplement preIter function.")

    def preIter(self):
        pass

    def postIter(self):
        pass

    def preEpoch(self):
        pass

    def postEpoch(self):
        pass

    def doForward(self):
        self.output = self.net(self.sample)

    def doLoss(self):
        self.loss = self.criterion(self.output, self.target)

    def doBackward(self):
        if self.conf.amp:
            self.scaler.scale(self.loss).backward()
        else:
            self.loss.backward()

    def doOptimize(self):
        if self.iter % self.conf.nominal_batch_factor != 0:
            return
        if self.conf.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        if self.conf.ema:
            self.updateEMA()

    def doLog(self):
        if self.step % self.conf.log_every != 0:
            return
        self.addScalar('{}/Loss'.format(self.phase), self.loss.item(), self.iter)
        self.addScalar('{}/LoadDataTime(secs/batch)'.format(self.phase), self.load_data_time.val, self.iter)
        self.addScalar('{}/DataCpu2GpuTime(secs/batch)'.format(self.phase), self.data_cpu2gpu_time.val, self.iter)
        self.addScalar('{}/TrainTime(secs/batch)'.format(self.phase), self.train_time.val, self.iter)
        LOG.logI('{}: [{}][{}/{}] [Loss:{}  Lr:{}]'.format(self.phase, self.epoch, self.step, self.loader_len,self.loss.item(),self.optimizer.param_groups[0]['lr']))

    def saveState(self, current_time):
        file_partial_name = '{}__acc_{}__epoch_{}__step_{}__lr_{}'.format(current_time, self.accuracy, self.epoch, self.step, self.optimizer.param_groups[0]['lr'])
        state_file = '{}/model__{}.pth'.format(self.output_dir, file_partial_name)
        checkpoint_file = '{}/checkpoint__{}.pth'.format(self.output_dir, file_partial_name)
        output_trace_file = '{}/trace__{}.pt'.format(self.output_dir, file_partial_name)
        output_script_file = '{}/script__{}.pt'.format(self.output_dir, file_partial_name)
        output_onnx_file = '{}/onnx__{}.onnx'.format(self.output_dir, file_partial_name)
        output_ncnn_file = '{}/ncnn__{}.bin'.format(self.output_dir, file_partial_name)
        output_coreml_file = '{}/coreml__{}.mlmodel'.format(self.output_dir, file_partial_name)
        #save state_dict
        net = self.ema if self.conf.ema else self.net
        torch.save(net.state_dict(), state_file)
        #save checkpoint
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'ema': self.ema.state_dict() if self.conf.ema else None,
            'scaler': self.scaler.state_dict() if self.conf.amp else None},  checkpoint_file)

        self.exportTorchViaTrace(self.sample, output_trace_file)
        self.exportTorchViaScript(output_script_file)
        self.exportONNX(output_onnx_file)
        self.exportNCNN(output_ncnn_file)
        self.exportCoreML(output_coreml_file)
        #tensorboard
        self.addScalar('{}/Accuracy'.format(self.phase), self.accuracy, self.iter)

    def processTrain(self):
        self.setTrainContext()
        self.step = 0
        LOG.logI('Phase {} started...'.format(self.phase))
        self.loader_len = len(self.loader)
        save_every = self.loader_len//self.conf.save_num
        save_list = list(range(0, self.loader_len + 1, save_every ))
        self.save_list = save_list[1:-1]
        LOG.logI('Model will be saved on step {} and the epoch end.'.format(self.save_list))
        self.addScalar('{}/LR'.format(self.phase), self.optimizer.param_groups[0]['lr'], self.epoch)
        self.preEpoch()
        self.train_time.reset()
        self.load_data_time.reset()
        self.data_cpu2gpu_time.reset()

        start = time.time()
        for i, (sample, target) in enumerate(self.loader):
            self.load_data_time.update(time.time() - start)
            self.step = i
            self.target = target
            self.sample = sample
            self.preIter()
            self.earlyIter()
            with autocast(enabled=self.conf.amp if self.conf.amp else False):
                self.doForward()
                self.doLoss()
            self.doBackward()
            self.doOptimize()
            self.doLog()
            self.postIter()
            self.iter += 1
            self.train_time.update(time.time() - start)
            if self.step in self.save_list:
                self.processVal()
                self.setTrainContext()
            start = time.time()

        self.addScalar('{}/TrainTime(hours/epoch)'.format(self.phase), round(self.train_time.sum / 3600, 2), self.epoch)
        self.addScalar('{}/AverageBatchTrainTime(secs/epoch)'.format(self.phase), self.train_time.avg, self.epoch)
        self.addScalar('{}/AverageBatchLoadDataTime(secs/epoch)'.format(self.phase), self.load_data_time.avg, self.epoch)
        self.addScalar('{}/AverageBatchDataCpu2GpuTime(secs/epoch)'.format(self.phase), self.data_cpu2gpu_time.avg, self.epoch)

        self.postEpoch()
        if self.scheduler:
            self.scheduler.step()

    def processVal(self, smoke=False):
        self.setValContext()
        LOG.logI('Phase {} started...'.format(self.phase))
        with torch.no_grad():
            self.preEpoch()
            for i, (sample, target) in enumerate(self.loader):
                self.target = target
                self.sample = sample
                self.preIter()
                self.earlyIter()
                self.doForward()
                self.doLoss()
                self.smokeTestForExport3rd()
                self.postIter()
                if smoke:
                    return
                LOG.logI('{}: [{}][{}/{}]'.format(self.phase, self.epoch, i, len(self.loader)))
            self.postEpoch()
        self.saveState(self.getTime())

    def processAccept(self):
        self.setValContext()

    def process(self):
        self.auditConfig()
        self.iter = 0
        epoch_start = self.epoch
        if self.conf.ema:
            self.ema_updates = self.epoch * len(self.train_loader) // self.conf.nominal_batch_factor
        self.processVal(smoke=True)
        self.optimizer.zero_grad()
        for epoch in range(epoch_start, self.conf.epoch_num):
            self.epoch = epoch
            LOG.logI('Epoch {} started...'.format(self.epoch))
            self.processTrain()
            self.processVal()
            self.processAccept()

    def __call__(self):
        self.process()

class DeepvacDDP(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(DeepvacDDP,self).__init__(deepvac_config)
        assert self.train_sampler is not None, "You should define self.train_sampler in DDP mode."

    def initDevice(self):
        super(DeepvacDDP, self).initDevice()
        parser = argparse.ArgumentParser(description='DeepvacDDP')
        parser.add_argument("--gpu", default=-1, type=int, help="gpu")
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        self.args = parser.parse_args()
        self.map_location = {'cuda:%d' % 0: 'cuda:%d' % self.args.rank}
        #in DDP, device may come from command line
        if self.args.gpu:
            self.device = torch.device(self.args.gpu)

        #os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.args.gpu)
        torch.cuda.set_device(self.args.gpu)

    def initDDP(self):
        LOG.logI("Start dist.init_process_group {} {}@{} on {}".format(self.conf.dist_url, self.args.rank, self.conf.world_size - 1, self.args.gpu))
        dist.init_process_group(backend='nccl', init_method=self.conf.dist_url, world_size=self.conf.world_size, rank=self.args.rank)
        #torch.cuda.set_device(self.args.gpu)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.args.gpu])
        LOG.logI("Finish dist.init_process_group {} {}@{} on {}".format(self.conf.dist_url, self.args.rank, self.conf.world_size - 1, self.args.gpu))

    def initTrainContext(self):
        self.initDDP()
        super(DeepvacDDP,self).initTrainContext()

    def initSummaryWriter(self):
        if self.args.rank != 0:
            return
        super(DeepvacDDP, self).initSummaryWriter()

    def preEpoch(self):
        self.train_sampler.set_epoch(self.epoch)

    def smokeTestForExport3rd(self):
        if self.args.rank != 0:
            return
        super(DeepvacDDP, self).smokeTestForExport3rd()

    def saveState(self, time):
        if self.args.rank != 0:
            return
        super(DeepvacDDP, self).saveState(time)

    def addScalar(self, tag, value, step):
        if self.args.rank != 0:
            return
        super(DeepvacDDP, self).addScalar(tag, value, step)

    def addImage(self, tag, image, step):
        if self.args.rank != 0:
            return
        super(DeepvacDDP, self).addImage(tag, image, step)

    @syszux_once
    def addGraph(self, input):
        if self.args.rank != 0:
            return
        self.writer.add_graph(self.net, input)

if __name__ == "__main__":
    from config import config as deepvac_config
    vac = Deepvac(deepvac_config)
    vac()
