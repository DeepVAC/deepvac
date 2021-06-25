from ..utils import LOG
from .base import DeepvacCast

class MnnCast(DeepvacCast):
    def auditConfig(self):
        if not self.config.model_dir:
            return False

        if not self.config.onnx2mnn:
            LOG.logE("You must set the onnx2mnn executable program path in config file. If you want to compile onnx2mnn tools, reference https://www.yuque.com/mnn/cn/cvrt_linux_mac  ", exit=True)
            return False
        return True

    def process(self, cast_output_file=None):
        output_mnn_file = self.config.model_dir
        if cast_output_file:
            output_mnn_file = '{}/mnn__{}.mnn'.format(self.trainer_config.output_dir, cast_output_file)
            self.config.model_dir = output_mnn_file
        
        LOG.logI("config.cast.MnnCast.model_dir found, save mnn model to {}...".format(self.config.model_dir))

        #to onnx, also set self.config.onnx_model_dir, self.config.onnx_input_names and self.config.onnx_output_names
        self.exportOnnx()

        cmd = self.config.onnx2mnn + " -f ONNX --modelFile " + self.config.onnx_model_dir + " --MNNModel " + output_mnn_file
        if self.config.save_static_model:
            cmd += " --saveStaticModel true"
        else:
            cmd += " --saveStaticModel false"
        rc, out_text, err_text = self.runCmd(cmd)

        if "" != err_text:
            LOG.logE(err_text + ". Error occured when export mnn model. ", exit=True)
        if "Converted Success" in out_text:
            LOG.logI("Pytorch model convert to Mnn model succeed, save mnn model file in {}.".format(output_mnn_file))
            return
        LOG.logE(out_text + ". Error occured when export mnn model. ", exit=True)