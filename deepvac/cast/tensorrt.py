from ..utils import LOG, syszux_once
from .base import DeepvacCast

TRT_LOGGER = None
@syszux_once
def initLog(trt):
    global TRT_LOGGER
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorrtCast(DeepvacCast):
    def auditConfig(self):
        if not self.config.model_dir:
            return False
    
        return True
    
    def process(self, cast_output_file=None):
        try:
            import tensorrt as trt
            initLog(trt)            
        except:
            LOG.logE("You must install tensorrt package if you want to convert pytorch to onnx. 1. Download Tensorrt7.2.3(for CUDA11.0) from https://developer.nvidia.com/tensorrt \
            2. unpack Tensorrt*.tar.gz  3. pip install tensorrt-x-cpx-none-linux_x86_64.whl in Tensorrt*(your_tensorrt_path)/python", exit=True)

        output_trt_file = self.config.model_dir
        if cast_output_file:
            output_trt_file = '{}/trt__{}.trt'.format(self.trainer_config.output_dir, cast_output_file)
            self.config.model_dir = output_trt_file
        
        LOG.logI("config.trt_model_dir found, save tensorrt model to {}...".format(self.config.model_dir))

        #to onnx, also set self.config.onnx_model_dir, self.config.onnx_input_names and self.config.onnx_output_names
        self.exportOnnx()

        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(self.config.onnx_model_dir, 'rb') as model:
                parser.parse(model.read())
            config = builder.create_builder_config()
            if self.config.onnx_dynamic_ax:
                profile = builder.create_optimization_profile()
                profile.set_shape(self.config.onnx_input_names[0], self.config.input_min_dims, self.config.input_opt_dims, self.config.input_max_dims)
                config.add_optimization_profile(profile)
            engine = builder.build_engine(network, config)
            with open(output_trt_file, "wb") as f:
                f.write(engine.serialize())
        LOG.logI("Pytorch model convert to TensorRT model succeed, save model in {}".format(output_trt_file))