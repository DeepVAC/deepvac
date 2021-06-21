from ..utils import is_ddp

if is_ddp:
    from .deepvac import DeepvacDDP as DeepvacTrain, Deepvac
else:
    from .deepvac import DeepvacTrain, Deepvac

from .config import AttrDict, interpret, newDict, new, fork
from .report import *