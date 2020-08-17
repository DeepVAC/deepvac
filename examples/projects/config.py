import torch 

from syszux_config import *
from torchvision import transforms


# global ************************************************************************************************************************************************************************************
config.lr = 1e-3
config.cls_num = 3
config.epoch_num = 50
config.num_workers = 4
config.input_size = (224, 224)
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train *************************************************************************************************************************************************************************************
config.train.shuffle = True
config.train.batch_size = 96
config.train.img_folder = "/gemfield/hostpv/nsfw/porn_cls3/train/"
config.train.transform_op = transforms.Compose([transforms.Resize(config.input_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                  ])

# val ***************************************************************************************************************************************************************************************
config.val.shuffle = True
config.val.batch_size = 100
config.val.img_folder = "/gemfield/hostpv/nsfw/porn_cls3/val/"
config.val.transform_op = transforms.Compose([transforms.Resize(config.input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])

# test **************************************************************************************************************************************************************************************
config.test.ds_name = "gemfield"
config.test.cls_to_idx = ['neutral', 'porn', 'sexy']
config.test.input_dir = "/gemfield/hostpv/nsfw/porn_cls3/test/"
config.test.model_path = "output/LTS_NSFW_train_standard/model:2020-08-17-10-05_acc:0.877159855021952_epoch:0_step:332_lr:0.001.pth"
