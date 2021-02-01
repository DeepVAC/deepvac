import sys

is_ddp = False
if '--rank' in sys.argv:
    is_ddp = True
    from .syszux_deepvac import DeepvacDDP as DeepvacTrain, Deepvac
else:
    from .syszux_deepvac import DeepvacTrain, Deepvac

from .syszux_log import LOG
from .syszux_config import AttrDict,config
from .syszux_loader import *
from .syszux_executor import *
from .syszux_aug import *
from .syszux_loss import *
from .syszux_report import *


__all__ = ['is_ddp', 'Deepvac', 'DeepvacTrain', 'LOG', 'AttrDict', 'config']
