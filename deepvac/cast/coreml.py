from ..utils import LOG
from .base import DeepvacCast

class CoremlCast(DeepvacCast):
    def __init__(self, deepvac_train_config):
        super(CoremlCast,self).__init__(deepvac_train_config)

    def auditConfig(self):
        if not self.config.coreml_model_dir:
            return False

        if not self.config.trace_model_dir and not self.config.script_model_dir:
            LOG.logE("CoreML converter now has dependency on TorchScript model, you need to enable config.train.trace_model_dir or config.train.script_model_dir", exit=True)

        if self.config.coreml_input_type not in [None, 'image','tensor']:
            LOG.logE("coreml input type must be {}".format([None, 'image','tensor']), exit=True)

        if self.config.coreml_input_type == 'image':
            LOG.logI("You are in coreml_input_type=image mod")
            if self.config.coreml_scale is None:
                LOG.logE("You need to set config.train.coreml_scale in config.py, e.g. config.train.coreml_scale = 1.0 / (0.226 * 255.0)", exit=True)
            
            if self.config.coreml_color_layout is None:
                LOG.logE("You need to set config.train.coreml_color_layout in config.py, e.g. config.train.coreml_color_layout = 'BGR' ", exit=True)

            if self.config.coreml_blue_bias is None:
                LOG.logE("You need to set config.train.coreml_blue_bias in config.py, e.g. config.train.coreml_blue_bias = -0.406 / 0.226 ", exit=True)

            if self.config.coreml_green_bias is None:
                LOG.logE("You need to set config.train.coreml_green_bias in config.py, e.g. config.train.coreml_green_bias = -0.456 / 0.226", exit=True)

            if self.config.coreml_red_bias is None:
                LOG.logE("You need to set config.train.coreml_red_bias in config.py, e.g. config.train.coreml_red_bias = -0.485 / 0.226 ", exit=True)

        return True

    def process(self, cast_output_file=None):
        try:
            import coremltools
        except:
            LOG.logE("You need to install coremltools package if you want to convert PyTorch to CoreML model. E.g. pip install --upgrade coremltools")
            return

        output_coreml_file = self.config.coreml_model_dir
        if cast_output_file:
            output_coreml_file = '{}/coreml__{}.mlmodel'.format(self.config.output_dir, cast_output_file)
            self.config.coreml_model_dir = output_coreml_file

        LOG.logI("config.coreml_model_dir found, save coreml model to {}...".format(self.config.coreml_model_dir))
        model = self.config.script_model_dir
        if self.config.script_model_dir is None:
            model = self.config.trace_model_dir
        #input mode
        if self.config.coreml_input_type == 'image':
            input = coremltools.ImageType(name="input",
                        shape=tuple(self.config.sample.shape),
                        scale=self.config.coreml_scale,
                        color_layout = self.config.coreml_color_layout,
                        bias = [self.config.coreml_blue_bias, self.config.coreml_green_bias, self.config.coreml_red_bias])
        else:
            input = coremltools.TensorType(name='input', shape=tuple(self.config.sample.shape))
        #convert
        coreml_model = coremltools.convert(model=model, inputs=[input], 
             classfier_config=self.config.coreml_classfier_config, minimum_deployment_target=self.config.coreml_minimum_deployment_target)

        # Set feature descriptions (these show up as comments in XCode)
        coreml_model.input_description["input"] = "Deepvac Model Input"
        coreml_model.output_description["classLabel"] = "Most likely image category"

        # Set model author name
        coreml_model.author = '"DeepVAC'

        # Set the license of the model
        coreml_model.license = "Deepvac Lincense"
        coreml_model.short_description = "Powered by DeepVAC"

        # Set a version for the model
        coreml_model.version = self.config.coreml_version if self.config.coreml_version else "1.0"
        # Save the CoreML model
        coreml_model.save(output_coreml_file)