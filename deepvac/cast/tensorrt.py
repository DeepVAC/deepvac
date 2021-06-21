from ..utils import LOG
from .base import DeepvacCast

class TensorrtCast(DeepvacCast):
    def auditConfig(self):
        if not self.config.model_dir:
            return False
        
        if self.config.enable_dynamic_input and self.config.onnx_dynamic_ax is None:
            LOG.logE("If you want to tensorrt support dynamic input, you must set onnx_dynamic_ax.", exit=True)
        
        return True

    def process(self, cast_output_file=None):
        try:
            import tensorrt as trt
        except:
            LOG.logE("You must install tensorrt package if you want to convert pytorch to onnx. 1. Download Tensorrt7.2.3(for CUDA11.0) from https://developer.nvidia.com/tensorrt \
            2. unpack Tensorrt*.tar.gz  3. pip install tensorrt-x-cpx-none-linux_x86_64.whl in Tensorrt*(your_tensorrt_path)/python", exit=True)
            return

        output_trt_file = self.config.model_dir
        if cast_output_file:
            output_trt_file = '{}/trt__{}.trt'.format(self.trainer_config.output_dir, cast_output_file)
            self.config.model_dir = output_trt_file
        
        LOG.logI("config.trt_model_dir found, save tensorrt model to {}...".format(self.config.model_dir))

        #to onnx, also set self.config.onnx_model_dir, self.config.onnx_input_names and self.config.onnx_output_names
        self.exportOnnx()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(trt_logger) as builder, builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, trt_logger) as parser:
            builder.max_workspace_size = 4 << 30
            builder.max_batch_size = 1
            with open(self.config.onnx_model_dir, 'rb') as model:
                parser.parse(model.read())
            config = builder.create_builder_config()
            if self.config.enable_dynamic_input:
                profile = builder.create_optimization_profile()
                profile.set_shape(self.config.onnx_input_names[0], self.config.input_min_dims, self.config.input_opt_dims, self.config.input_max_dims)
                config.add_optimization_profile(profile)
            engine = builder.build_engine(network, config)
            with open(output_trt_file, "wb") as f:
                f.write(engine.serialize())
        LOG.logI("Pytorch model convert to TensorRT model succeed, save model in {}".format(output_trt_file))