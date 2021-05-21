import subprocess
import tempfile
import copy
import torch
from torch.quantization import quantize_dynamic_jit, per_channel_dynamic_qconfig
from torch.quantization import get_default_qconfig, quantize_jit
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
import torch.nn as nn
from ..core.config import AttrDict
from ..utils import LOG

class DeepvacCast(object):
    def __init__(self, deepvac_config):
        self.deepvac_cast_config = deepvac_config.cast
        self.deepvac_core_config = deepvac_config.core
        self.initConfig()
        self.proceed = False
        self.auditFinalConfig()

    def initConfig(self):
        if self.name() not in self.deepvac_cast_config.keys():
            self.deepvac_cast_config[self.name()] = AttrDict()
        self.config = self.deepvac_cast_config[self.name()]

    def addUserConfig(self, config_name, user_give=None, developer_give=None, is_user_mandatory=False):
        module_name = 'config.cast.{}'.format(self.name())
        return addUserConfig(module_name, config_name, user_give=user_give, developer_give=developer_give, is_user_mandatory=is_user_mandatory)

    def setAttr(self, k, v):
        self.config[k] = v

    def getAttr(self,k):
        return self.config[k]

    def auditConfig(self):
        LOG.logW("You should reimplement auditConfig in your subclass {}.".format(self.name()))

    def name(self):
        return self.__class__.__name__

    def auditFinalConfig(self):
        if not self.auditConfig():
            return
        if self.deepvac_core_config.net is None:
            LOG.logE("You must set config.core.net in config.py", exit=True)

        if self.deepvac_core_config.sample is None:
            LOG.logE("You must set config.core.sample, in general, this is done by Deepvac Framework.", exit=True)
        
        self.net = copy.deepcopy(self.deepvac_core_config.ema if self.deepvac_core_config.ema else self.deepvac_core_config.net)
        if self.deepvac_core_config.cast2cpu:
            self.net.cpu()
            self.deepvac_core_config.sample = self.deepvac_core_config.sample.to('cpu')
        
        self.proceed = True
        
    def process(self, cast_output_file=None):
        LOG.logE("You must reimplement process() in your subclass {}.".format(self.name()), exit=True)

    def exportOnnx(self):
        if self.config.onnx_input_names is None:
            self.config.onnx_input_names = ["input"]
        
        if self.config.onnx_output_names is None:
            self.config.onnx_output_names = ["output"]

        if not self.config.onnx_model_dir:
            f = tempfile.NamedTemporaryFile(delete=False)
            self.config.onnx_model_dir = f.name

        torch.onnx.export(self.net, self.deepvac_core_config.sample, self.config.onnx_model_dir, 
            input_names=self.config.onnx_input_names, output_names=self.config.onnx_output_names, 
            dynamic_axes=self.config.onnx_dynamic_ax, opset_version=self.config.onnx_version, export_params=True)
        LOG.logI("Pytorch model convert to ONNX model succeed, save model in {}".format(self.config.onnx_model_dir))

    def __call__(self, cast_output_file=None):
        if not self.proceed:
            return
        self.process(cast_output_file)

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for sample, target in data_loader:
            model(sample)
            
class ScriptModel(object):
    def __init__(self, deepvac_core_config, output_file, backend = 'fbgemm'):
        self.deepvac_core_config = deepvac_core_config
        self.input_net = copy.deepcopy(self.deepvac_core_config.ema if self.deepvac_core_config.ema else self.deepvac_core_config.net)
        self.input_net.to(self.deepvac_core_config.sample.device)
        self.input_net.eval()
        self.output_file = output_file
        self.backend = backend
        self.dq_output_file = '{}.dq'.format(output_file)
        self.sq_output_file = '{}.sq'.format(output_file)
        self.d_qconfig_dict = {'': per_channel_dynamic_qconfig}
        self.s_qconfig_dict = {'': get_default_qconfig(self.backend) }

    def _freeze_jit(self, model):
        LOG.logE("You must reimplement _freeze_jit() in ScriptModel subclass.", exit=True)
    
    def _jit(self, model):
        LOG.logE("You must reimplement _jit() in ScriptModel subclass.", exit=True)

    def getCalibrateLoader(self):
        loader = self.deepvac_core_config.test_loader if self.deepvac_core_config.is_forward_only else self.deepvac_core_config.val_loader
        if loader is None:
            LOG.logE("You enabled config.static_quantize_dir, but didn't provide config.test_loader in forward-only mode, or self.val_loader in train mode.", exit=True)
        return loader

    def save(self):
        with torch.no_grad():
            torch.jit.save(self._freeze_jit(self.input_net), self.output_file)

    def saveDQ(self):
        LOG.logI("Pytorch model dynamic quantize starting, will save model in {}".format(self.dq_output_file))
        with torch.no_grad():
            quantized_model = quantize_dynamic_jit(self._jit(), self.d_qconfig_dict)
            torch.jit.save(quantized_model, self.dq_output_file)
        LOG.logI("Pytorch model dynamic quantize succeeded, saved model in {}".format(self.dq_output_file))

    def saveSQ(self):
        loader = self.getCalibrateLoader()
        LOG.logI("Pytorch model static quantize starting, will save model in {}".format(self.sq_output_file))
        with torch.no_grad():
            quantized_model = quantize_jit(self._jit(), self.s_qconfig_dict, calibrate, [loader], inplace=False, debug=False)
            torch.jit.save(quantized_model, self.sq_output_file)
        LOG.logI("Pytorch model static quantize succeeded, saved model in {}".format(self.sq_output_file))

class SaveModelByTrace(ScriptModel):
    def _freeze_jit(self, model):
        return torch.jit.freeze(torch.jit.trace(model, self.deepvac_core_config.sample).eval() )
    
    def _jit(self, model):
        return torch.jit.trace(model, self.deepvac_core_config.sample).eval()

class SaveModelByScript(ScriptModel):
    def _freeze_jit(self, model):
        return torch.jit.freeze(torch.jit.script(model).eval() )

    def _jit(self, model):
        return torch.jit.script(model).eval()

class FXQuantize(ScriptModel):
    def saveDQ(self):
        qconfig_dict = {
            "object_type": [
                (nn.Embedding, float_qparams_weight_only_qconfig),
                (nn.LSTM, default_dynamic_qconfig),
                (nn.Linear, default_dynamic_qconfig)
            ]
        }
        prepared_model = prepare_fx(self.input_net, qconfig_dict)
        quantized_model = convert_fx(prepared_model)
        torch.jit.save(self._freeze_jit(quantized_model), self.dq_output_file)

    def saveSQ(self):
        loader = self.getCalibrateLoader()
        prepared_model = prepare_fx(self.input_net, self.s_qconfig_dict)
        #print(prepared_model.graph)
        calibrate(prepared_model, loader)
        quantized_model = convert_fx(prepared_model)
        torch.jit.save(self._freeze_jit(quantized_model), self.sq_output_file)

class EagerQuantize(ScriptModel):
    def saveDQ(self):
        quantized_model = torch.quantization.quantize_dynamic(self.input_net, inplace=False)
        torch.jit.save(self._freeze_jit(quantized_model), self.dq_output_file)

    def saveSQ(self):
        loader = self.getCalibrateLoader()
        fused_model = auto_fuse_model(self.input_net)
        fused_model.qconfig = torch.quantization.get_default_qconfig(self.backend)
        prepared_model = torch.quantization.prepare(fused_model, inplace=False)
        calibrate(prepared_model, loader)
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        torch.jit.save(self._freeze_jit(quantized_model), self.sq_output_file)

class FXQuantizeAndScript(FXQuantize, SaveModelByScript):
    pass

class FXQuantizeAndTrace(FXQuantize, SaveModelByTrace):
    pass

class EagerQuantizeAndScript(EagerQuantize, SaveModelByScript):
    pass

class EagerQuantizeAndTrace(EagerQuantize, SaveModelByTrace):
    pass
