import os
import time
from ..utils import LOG
from .base import DeepvacCast

class TNNCast(DeepvacCast):
    def auditConfig(self):
        if not self.config.model_dir:
            return False

        return True

    def process(self, cast_output_file=None):
        try:
            import onnx2tnn
        except:
            LOG.logE("You must install onnx2tnn package if you want to convert pytorch to tnn. \
                    1. compile onnx2tnn, reference https://github.com/Tencent/TNN/blob/master/doc/en/user/onnx2tnn_en.md#compile. \
                    2. copy <path-to-tnn>/tools/onnx2tnn/onnx-converter/*.so to your python site-packages locate(you can use pip3 -V show where your packages are located).", exit=True)
            return

        output_tnn_file = self.config.model_dir
        if cast_output_file:
            output_tnn_file = '{}/tnn__{}.tnnmodel'.format(self.trainer_config.output_dir, cast_output_file)
            self.config.model_dir = output_tnn_file
        self.config.arch_dir = '{}.tnnproto'.format(output_tnn_file)
        
        LOG.logI("config.cast.TNNCast.model_dir found, save tnn model to {}...".format(self.config.model_dir))

        #to onnx, also set self.config.onnx_model_dir, self.config.onnx_input_names and self.config.onnx_output_names
        self.exportOnnx()
        if self.config.optimize:
            self.onnxSimplify(input_shapes=input_shapes_, perform_optimization=False)
        
        output_dir = os.path.dirname(self.config.model_dir)
        command = "python3 -c \"import onnx2tnn;onnx2tnn.convert(\\\""+self.config.onnx_model_dir+"\\\",\\\""+output_dir+"\\\",\\\"v1.0\\\",\\\""+time.strftime("%Y%m%d %H:%M:%S", time.localtime())+"\\\",0,"+ str(1 if self.config.optimize else 0)+",\\\"\\\")\""
        rc, out_text, err_text = self.runCmd(command)
    
        if rc != 0:
            LOG.logE(err_text + ". Error occured when export tnn model.", exit=True)

        #rename tnnmodel and tnnproto, cuz we can't control tnn model name
        prefix = os.path.splitext(os.path.basename(self.config.onnx_model_dir))[0]
        os.rename(os.path.join(output_dir, prefix+".tnnmodel"), self.config.model_dir)
        os.rename(os.path.join(output_dir, prefix+".tnnproto"), self.config.arch_dir)
        LOG.logI("Pytorch model convert to TNN model succeed, save tnn model file in {}, save tnn proto file in {}".format(self.config.model_dir, self.config.arch_dir))