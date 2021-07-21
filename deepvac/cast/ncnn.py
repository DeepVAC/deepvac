import os
from ..utils import LOG
from .base import DeepvacCast

class NcnnCast(DeepvacCast):
    def auditConfig(self):
        if not self.config.model_dir:
            return False
        
        self.config.onnx2ncnn = self.addUserConfig("onnx2ncnn", self.config.onnx2ncnn, "/bin/onnx2ncnn")
        exist = os.path.exists(self.config.onnx2ncnn)
        if not exist:
            LOG.logE("The config.cast.NcnnCast.onnx2ncnn is invalid. We try to use default setting of config.cast.NcnnCast.onnx2ncnn is failed too. \
                    If you want to compile onnx2ncnn tools, reference https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86 ", exit=True)
            return False
        
        return True

    def process(self, cast_output_file=None):
        output_ncnn_file = self.config.model_dir
        
        if cast_output_file:
            output_ncnn_file = '{}/ncnn__{}.bin'.format(self.trainer_config.output_dir, cast_output_file)
            self.config.model_dir = output_ncnn_file
        self.config.arch_dir = '{}.param'.format(output_ncnn_file)
        LOG.logI("config.cast.NcnnCast.model_dir found, save ncnn model to {}...".format(self.config.model_dir))

        #to onnx, also set self.config.onnx_model_dir, self.config.onnx_input_names and self.config.onnx_output_names
        self.exportOnnx()

        cmd = "{} {} {} {}".format(self.config.onnx2ncnn, self.config.onnx_model_dir, self.config.arch_dir, output_ncnn_file)
        rc, out_text, err_text = self.runCmd(cmd)
        if err_text == "":
            LOG.logI("Pytorch model convert to NCNN model succeed, save ncnn param file in {}, save ncnn bin file in {}.".format(self.config.arch_dir, output_ncnn_file))
            return

        LOG.logE(err_text + ". Error occured when export ncnn model. Deepvac try to simplify the model first.")

        try:
            import onnx
            from onnxsim import simplify
        except:
            LOG.logE("You must install onnx and onnxsim package first if you want to convert pytorch to ncnn.", exit=True)
        
        dynamic_input_shape = False
        input_shapes = None
        if self.config.onnx_dynamic_ax:
            dynamic_input_shape = True
            input_shapes = {self.config.onnx_input_names[0]: self.trainer_config.sample.shape}

        self.onnxSimplify(check_n=3, perform_optimization=True, skip_fuse_bn=True,  skip_shape_inference=False, dynamic_input_shape=dynamic_input_shape, input_shapes=input_shapes)
        
        rc, out_text, err_text = self.runCmd(cmd)
        if err_text != "":
            LOG.logE(err_text + ". we can't guarantee generate model is right.", exit=True)
        
        if not os.path.isfile(output_ncnn_file):
            LOG.logE("Error: ncnn model not generated due to internal error!", exit=True)

        LOG.logI("Pytorch model convert to NCNN model succeed, save ncnn param file in {}, save ncnn bin file in {}".format(self.config.arch_dir, output_ncnn_file))