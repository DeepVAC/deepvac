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
config.feature = AttrDict()
config.aug = AttrDict()

## ------------------ common ------------------
config.train.device = "cuda"
config.train.output_dir = "output"
config.train.log_dir = "log"
config.train.log_every = 10
config.train.disable_git = False
config.train.cast2cpu = True

## ------------------ ddp --------------------
config.train.dist_url = "tcp://localhost:27030"
config.train.world_size = 2

## ------------------ optimizer  ------------------
config.train.lr = 0.01
config.train.lr_step = None
config.train.lr_factor = 0.2703
config.train.momentum = 0.9
config.train.nesterov = False
config.train.weight_decay = None

## -------------------- loader ------------------
config.train.num_workers = 3
config.train.nominal_batch_factor = 1

## ------------------- train ------------------
config.train.train_batch_size = 128
config.train.epoch_num = 30
#model save number duriong an epoch
config.train.save_num = 5
config.train.milestones = [2,4,6,8,10]
config.train.checkpoint_suffix = ''
#
config.train.acc = 0

## ------------------ val ------------------
config.train.val_batch_size = None

