from ..utils import LOG
from .base import DeepvacCast

class CoremlCast(DeepvacCast):
    def auditConfig(self):
        if not self.config.model_dir:
            return False

        if not self.deepvac_cast_config.TraceCast.model_dir and not self.deepvac_cast_config.ScriptCast.model_dir:
            LOG.logE("CoreML converter now has dependency on TorchScript model, you need to enable config.cast.TraceCast.model_dir or config.cast.ScriptCast.model_dir", exit=True)

        if self.config.input_type not in [None, 'image','tensor']:
            LOG.logE("coreml input type must be {}".format([None, 'image','tensor']), exit=True)

        if self.config.input_type == 'image':
            LOG.logI("You are in coreml_input_type=image mod")
            if self.config.scale is None:
                LOG.logE("You need to set config.cast.CoremlCast.scale in config.py, e.g. config.cast.CoremlCast.scale = 1.0 / (0.226 * 255.0)", exit=True)
            
            if self.config.color_layout is None:
                LOG.logE("You need to set config.cast.CoremlCast.color_layout in config.py, e.g. config.cast.CoremlCast.color_layout = 'BGR' ", exit=True)

            if self.config.blue_bias is None:
                LOG.logE("You need to set config.cast.CoremlCast.blue_bias in config.py, e.g. config.cast.CoremlCast.blue_bias = -0.406 / 0.226 ", exit=True)

            if self.config.green_bias is None:
                LOG.logE("You need to set config.cast.CoremlCast.green_bias in config.py, e.g. config.cast.CoremlCast.green_bias = -0.456 / 0.226", exit=True)

            if self.config.red_bias is None:
                LOG.logE("You need to set config.cast.CoremlCast.red_bias in config.py, e.g. config.cast.CoremlCast.red_bias = -0.485 / 0.226 ", exit=True)

        return True

    def process(self, cast_output_file=None):
        try:
            import coremltools
        except:
            LOG.logE("You need to install coremltools package if you want to convert PyTorch to CoreML model. E.g. pip install --upgrade coremltools")
            return

        output_coreml_file = self.config.model_dir
        if cast_output_file:
            output_coreml_file = '{}/coreml__{}.mlmodel'.format(self.trainer_config.output_dir, cast_output_file)
            self.config.model_dir = output_coreml_file

        LOG.logI("config.cast.CoremlCast.model_dir found, save coreml model to {}...".format(self.config.model_dir))
        model = self.deepvac_cast_config.ScriptCast.model_dir
        if self.deepvac_cast_config.ScriptCast.model_dir is None:
            model = self.deepvac_cast_config.TraceCast.model_dir
        #input mode
        if self.config.input_type == 'image':
            input = coremltools.ImageType(name="input",
                        shape=tuple(self.trainer_config.sample.shape),
                        scale=self.config.scale,
                        color_layout = self.config.color_layout,
                        bias = [self.config.blue_bias, self.config.green_bias, self.config.red_bias])
        else:
            input = coremltools.TensorType(name='input', shape=tuple(self.trainer_config.sample.shape))
        #convert
        coreml_model = coremltools.convert(model=model, inputs=[input], 
             classfier_config=self.config.classfier_config, minimum_deployment_target=self.config.minimum_deployment_target)

        # Set feature descriptions (these show up as comments in XCode)
        coreml_model.input_description["input"] = "Deepvac Model Input"

        # Set model author name
        coreml_model.author = '"DeepVAC'

        # Set the license of the model
        coreml_model.license = "Deepvac Lincense"
        coreml_model.short_description = "Powered by DeepVAC"

        # Set a version for the model
        coreml_model.version = self.config.version if self.config.version else "1.0"
        # Save the CoreML model
        coreml_model.save(output_coreml_file)
