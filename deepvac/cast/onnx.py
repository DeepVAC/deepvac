from ..utils import LOG
from .base import DeepvacCast

class OnnxCast(DeepvacCast):
    def __init__(self, deepvac_train_config):
        super(OnnxCast,self).__init__(deepvac_train_config)

    def auditConfig(self):
        if not self.config.onnx_model_dir:
            return False
        return True

    def process(self, cast_output_file=None):
        output_onnx_file = self.config.onnx_model_dir
        if cast_output_file:
            output_onnx_file = '{}/onnx__{}.onnx'.format(self.config.output_dir, cast_output_file)
            self.config.onnx_model_dir = output_onnx_file
        LOG.logI("config.onnx_model_dir found, save onnx model to {}...".format(output_onnx_file))
        self.exportOnnx()