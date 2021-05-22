import copy
primitive = (int, float, str, bool, list, dict, tuple, set)

class AttrDict(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, memo=None):
        try:
            ad = AttrDict(copy.deepcopy(dict(self), memo=memo))
        except Exception as e:
            print("Warning, issue happened in clone() API. Error: {}".format(str(e)))
            ad = AttrDict()
        return ad

    def clone(self):
        return copy.deepcopy(self)

def interpret(name):
    fields = name.split('.')
    c = AttrDict()
    if len(fields) < 2:
        return c
    cd = c
    for f in fields[1:]:
        cd[f] = AttrDict()
        cd=cd[f]
    return c

def newDict():
    return AttrDict()

def new():
    config = AttrDict()
    config.core = AttrDict()
    config.feature = AttrDict()
    config.aug = AttrDict()
    config.cast = AttrDict()
    config.backbones = AttrDict()
    config.loss = AttrDict()
    config.datasets = AttrDict()
    return config

def fork(deepvac_config, field=['aug','datasets']):
    if not isinstance(field, list):
        field = [field]
    c = new()
    for f in field:
        if f not in deepvac_config.keys():
            print("ERROR: deepvac fork found unsupport field: {}".format(f))
            return None
        c[f] = deepvac_config[f].clone()
    return c

config = new()

## ------------------ common ------------------
config.core.device = "cuda"
config.core.output_dir = "output"
config.core.log_dir = "log"
config.core.log_every = 10
config.core.disable_git = False
config.core.cast2cpu = True
config.core.model_reinterpret_cast=False
config.core.cast_state_dict_strict=True

## ------------------ ddp --------------------
config.core.dist_url = "tcp://localhost:27030"
config.core.world_size = 2

## ------------------ optimizer  ------------------
config.core.lr = 0.01
config.core.lr_step = None
config.core.lr_factor = 0.2703
config.core.momentum = 0.9
config.core.nesterov = False
config.core.weight_decay = None

## -------------------- loader ------------------
config.core.num_workers = 3
config.core.nominal_batch_factor = 1

## ------------------- train ------------------
config.core.train_batch_size = 128
config.core.epoch_num = 30
#model save number duriong an epoch
config.core.save_num = 5
config.core.milestones = [2,4,6,8,10]
config.core.checkpoint_suffix = ''
#
config.core.acc = 0
config.core.phase = 'TRAIN'

## ------------------ val ------------------
config.core.val_batch_size = None

