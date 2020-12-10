import torch 
from deepvac.syszux_config import *
from torchvision import transforms

# global
config.disable_git = True
config.lr = 1e-3
config.cls_num = 3
config.epoch_num = 50
config.num_workers = 4
config.input_size = (224, 224)
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train
config.train.shuffle = True
config.train.batch_size = 96
config.train.img_folder = "/gemfield/hostpv/nsfw/train/"
config.train.transform_op = transforms.Compose([transforms.Resize(config.input_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                  ])

# val
config.val.shuffle = True
config.val.batch_size = 100
config.val.img_folder = "/gemfield/hostpv/nsfw/val/"
config.val.transform_op = transforms.Compose([transforms.Resize(config.input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])

# test
config.model_path = '/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
config.test.ds_name = "gemfield"
config.test.input_dir = "/gemfield/hostpv/nsfw/test/"
config.test.transform_op = transforms.Compose([transforms.Resize(config.input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])
