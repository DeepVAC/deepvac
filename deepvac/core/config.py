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

def new(trainer=None):
    config = AttrDict()
    config.core = AttrDict()
    config.feature = AttrDict()
    config.aug = AttrDict()
    config.cast = AttrDict()
    config.backbones = AttrDict()
    config.loss = AttrDict()
    config.datasets = AttrDict()
    if trainer is None:
        return config
    
    config.core[trainer] = AttrDict()
    ## ------------- common ------------------
    config.core[trainer].device = "cuda"
    config.core[trainer].output_dir = "output"
    config.core[trainer].log_dir = "log"
    config.core[trainer].log_every = 10
    config.core[trainer].disable_git = False
    config.core[trainer].cast2cpu = True
    config.core[trainer].model_reinterpret_cast=False
    config.core[trainer].cast_state_dict_strict=True
    config.core[trainer].model_path_omit_keys=[]
    config.core[trainer].net_omit_keys_strict=[]

    ## ----------------- ddp --------------------
    config.core[trainer].dist_url = "tcp://localhost:27030"
    config.core[trainer].world_size = 2
    config.core[trainer].shuffle = False

    ## ------------------- loader ------------------
    config.core[trainer].num_workers = 3
    config.core[trainer].nominal_batch_factor = 1

    ## ------------------ train ------------------
    config.core[trainer].train_batch_size = 128
    config.core[trainer].epoch_num = 30
    #model save number during an epoch
    config.core[trainer].save_num = 5
    config.core[trainer].checkpoint_suffix = ''

    ## ----------------- val ------------------
    config.core[trainer].val_batch_size = None
    config.core[trainer].acc = 0

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