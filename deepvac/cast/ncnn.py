import os
import subprocess
from ..utils import LOG
from .base import DeepvacCast

class NcnnCast(DeepvacCast):
    def auditConfig(self):
        if not self.config.model_dir:
            return False

        if not self.config.onnx2ncnn:
            LOG.logE("You must set the onnx2ncnn executable program path in config file. If you want to compile onnx2ncnn tools, reference https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86 ", exit=True)
            return False
        return True

    def process(self, cast_output_file=None):
        output_ncnn_file = self.config.model_dir
        if cast_output_file:
            output_ncnn_file = '{}/ncnn__{}.bin'.format(self.config.output_dir, cast_output_file)
            self.config.model_dir = output_ncnn_file
        self.config.arch_dir = '{}.param'.format(output_ncnn_file)
        LOG.logI("config.cast.NcnnCast.model_dir found, save ncnn model to {}...".format(self.config.model_dir))

        #to onnx, also set self.config.onnx_model_dir, self.config.onnx_input_names and self.config.onnx_output_names
        self.exportOnnx()

        cmd = self.config.onnx2ncnn + " " + self.config.onnx_model_dir + " " + self.config.arch_dir + " " + output_ncnn_file
        pd = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pd.stderr.read() == b"":
            LOG.logI("Pytorch model convert to NCNN model succeed, save ncnn param file in {}, save ncnn bin file in {}.".format(self.config.arch_dir, output_ncnn_file))
            return

        LOG.logE(pd.stderr.read() + b". Error occured when export ncnn model. Deepvac try to simplify the model first.")
        try:
            import onnx
            from onnxsim import simplify
        except:
            LOG.logE("You must install onnx and onnxsim package first if you want to convert pytorch to ncnn.", exit=True)

        model_op, check_ok = simplify(self.config.onnx_model_dir, check_n=3, perform_optimization=True, skip_fuse_bn=True,  skip_shape_inference=False)
        onnx.save(model_op, self.config.onnx_model_dir)
        if not check_ok:
            LOG.logE("Maybe something wrong when simplify the model, we can't guarantee generate model is right.")
        else:
            LOG.logI("Simplify model succeed")
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pd.stderr.read() != b"":
            LOG.logE(pd.stderr.read() + b". we can't guarantee generate model is right.", exit=True)
        
        if not os.path.isfile(output_ncnn_file):
            LOG.logE("Error: ncnn model not generated due to internal error!", exit=True)

        LOG.logI("Pytorch model convert to NCNN model succeed, save ncnn param file in {}, save ncnn bin file in {}".format(self.config.arch_dir, output_ncnn_file))
