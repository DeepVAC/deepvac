from ..utils import LOG
from .base import DeepvacCast, FXQuantizeAndTrace

class TraceCast(DeepvacCast):
    def auditConfig(self): 
        if not self.config.model_dir:
            if self.config.dynamic_quantize_dir:
                LOG.logE("You must set config.cast.TraceCast.model_dir when you trying to enable config.cast.TraceCast.dynamic_quantize_dir.", exit=True)
            if self.config.static_quantize_dir:
                LOG.logE("You must set config.cast.TraceCast.model_dir when you trying to enable config.cast.TraceCast.static_quantize_dir.", exit=True)
            return False

        if self.config.static_quantize_dir:
            if self.trainer_config.is_forward_only and self.trainer_config.test_loader is None:
                LOG.logE("You must set config.core.test_loader in config.py when static_quantize_dir is enabled in TEST.", exit=True)
            if not self.trainer_config.is_forward_only and self.trainer_config.val_loader is None:
                LOG.logE("You must set config.core.val_loader in config.py when static_quantize_dir is enabled in TRAIN.", exit=True)

        return True

    def process(self, cast_output_file=None):
        output_trace_file = self.config.model_dir
        if cast_output_file:
            output_trace_file = '{}/trace_{}.pt'.format(self.trainer_config.output_dir, cast_output_file)

        LOG.logI("config.trace_model_dir found, save trace model to {}...".format(output_trace_file))
        trace_model = FXQuantizeAndTrace(self.trainer_config, output_trace_file)
        trace_model.save()

        if self.config.dynamic_quantize_dir:
            LOG.logI("You have enabled config.cast.TraceCast.dynamic_quantize_dir, will dynamic quantize the model...")
            trace_model.saveDQ()

        if self.config.static_quantize_dir:
            LOG.logI("You have enabled config.cast.TraceCast.static_quantize_dir, will static quantize the model...")
            trace_model.saveSQ()