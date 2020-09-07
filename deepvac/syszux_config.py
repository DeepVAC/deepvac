class AttrDict(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

config = AttrDict()
config.train = AttrDict()
config.val = AttrDict()
config.test = AttrDict()

## ------------------ common ------------------
config.device = "cuda"
config.output_dir = "output"
config.log_dir = "log"
config.log_every = 10
config.disable_git = False

## ------------------ ddp --------------------
config.dist_url = "tcp://localhost:27030"
config.world_size = 1

## ------------------ optimizer  ------------------
config.lr = 0.01
config.lr_step = None
config.lr_factor = 0.2703
config.momentum = 0.9
config.nesterov = False
config.weight_decay = None

## -------------------- loader ------------------
config.num_workers = 3

## ------------------- train ------------------
config.train.batch_size = 128
config.epoch_num = 30
#model save number duriong an epoch
config.save_num = 5
config.milestones = [2,4,6,8,10]
config.checkpoint_suffix = ''

## ------------------ val ------------------
config.val.batch_size = None


