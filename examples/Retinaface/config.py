import torch 
from deepvac.syszux_config import *
from torchvision import transforms

# global
config.disable_git = True
config.cls_num = 2
config.input_size = 640
config.epoch_num = 250
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.output_dir = 'output'

config.num_workers = 4
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 1e-3
config.gamma = 0.1
config.confidence_threshold = 0.02
config.nms_threshold = 0.4
config.top_k = 5000
config.keep_top_k = 750

config.cfg_mobilenet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': (640, 640),
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64    
}

config.cfg_resnet = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': (840, 840),
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256    
}
config.network = 'resnet50' # or 'moblilenet0.25'
config.cfg = config.cfg_mobilenet if config.network=='moblilenet0.25' else config.cfg_resnet
# train
config.train.shuffle = True
config.train.batch_size = 8
config.train.img_folder = '/gemfield/hostpv/wangyuhang/new/data/widerface/train/images'
config.train.label_path = '/gemfield/hostpv/wangyuhang/new/data/widerface/train/label.txt'

# val
config.val.shuffle = True
config.val.img_folder = '/gemfield/hostpv/wangyuhang/new/data/widerface/val/images'
config.val.batch_size = 1
config.val.label_path = '/gemfield/hostpv/wangyuhang/new/data/wider_face_split/label_val_.txt'
