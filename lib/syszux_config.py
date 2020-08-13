class AttrDict(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

config = AttrDict()

## ------------------ common ------------------
config.device = "cuda"
config.output_dir = "./output"
config.log_dir = "./log"
config.log_every = 10

## ------------------ ddp --------------------
config.dist_url = "tcp://localhost:27030"
config.world_size = 1

## ------------------ optimizer  ------------------
config.lr = 0.01
config.lr_step = None
config.lr_factor = 0.1
config.momentum = 0.9
config.nesterov = False
config.weight_decay = None

## -------------------- loader ------------------
config.num_workers = 3

## ------------------- train ------------------
config.train_batch_size = 128
config.epoch_num = 30
#model save number duriong an epoch
config.save_num = 5

## ------------------ val/test ------------------
config.val_batch_size = None

## ------------------ export ------------------
config.input_shape = {
    "channel": 3,
    "height": 112,
    "width": 112
}
#onnx2ncnn executable program path
config.onnx2ncnn = ""

config.onnx_output_model_path = ""
config.ncnn_param_output_path = ""
config.ncnn_bin_output_path = ""
