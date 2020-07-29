import sys
sys.path.append("lib")
sys.path.append("../lib")
sys.path.append("../../lib")

from config import config as deepvac_config
from syszux_synthesis_factory import SynthesisFactory

synthesis = SynthesisFactory()

#pure = synthesis.get('SynthesisTextPure')(deepvac_config.text)
#pure()

from_video = synthesis.get('SynthesisTextFromVideo')(deepvac_config.text)
from_video()

#from_images = synthesis.get('SynthesisTextFromImage')(deepvac_config.text)
#from_images()

