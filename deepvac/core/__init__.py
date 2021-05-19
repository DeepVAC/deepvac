from ..utils import is_ddp

if is_ddp:
    from .deepvac import DeepvacDDP as DeepvacTrain, Deepvac
else:
    from .deepvac import DeepvacTrain, Deepvac

from .config import config, AttrDict, fork, new, newDict
from .report import *