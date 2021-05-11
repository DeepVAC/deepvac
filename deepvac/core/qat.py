from ..utils.ddp import is_ddp
if is_ddp:
    from .deepvac import DeepvacDDP as DeepvacTrain
else:
    from .deepvac import DeepvacTrain

class DeepvacQAT(DeepvacTrain):
    pass