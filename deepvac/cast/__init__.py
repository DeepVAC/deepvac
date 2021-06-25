from .trace import TraceCast
from .script import ScriptCast
from .onnx import OnnxCast
from .coreml import CoremlCast
from .ncnn import NcnnCast
from .tensorrt import TensorrtCast
from .mnn import MnnCast
from .tnn import TnnCast

caster_list = ['Trace', 'Script', 'Onnx', 'Coreml', 'Ncnn', 'Tensorrt', 'Mnn', 'Tnn']
def export3rd(trainer_config, cast_config, output_file=None):
    for caster_name in caster_list:
        caster = eval('{}Cast(trainer_config, cast_config)'.format(caster_name))
        caster(output_file)
        
