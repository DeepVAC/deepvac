import sys
sys.path.append("..")
from syszux_synthesis import SynthesisTextPure
from syszux_synthesis import SynthesisTextFromVideo
from conf import *

pure = SynthesisTextPure(config.text)
pure()

synthesis = SynthesisTextFromVideo(config.text)
synthesis()
