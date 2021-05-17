from .trace import TraceCast
from .script import ScriptCast
from .onnx import OnnxCast
from .coreml import CoremlCast
from .ncnn import NcnnCast
from .tensorrt import TensorrtCast

caster_list = ['Trace', 'Script', 'Onnx', 'Coreml', 'Ncnn', 'Tensorrt']
def export3rd(deepvac_config, output_file=None):
    for caster_name in caster_list:
        caster = eval('{}Cast(deepvac_config)'.format(caster_name))
        caster(output_file)
        
