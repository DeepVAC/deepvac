import os
from deepvac.syszux_config import *
##########classifier report########
config.cls = AttrDict()
config.cls.file_path = '/ your output path /'
config.cls.cls_num = 90219 # numbers of idx
config.cls.db_paths = [] # your db_paths list
config.cls.map_locs = [{'cuda:0':'cuda:1'}, {'cuda:0':'cuda:1'}, {'cuda:1':'cuda:0'}, {'cuda:1':'cuda:0'}] # your device map_locs list
config.cls.np_paths = [] # your np_paths list
##########classifier_faiss report########
config.cls_faiss = AttrDict()
config.cls_faiss.cls_num = 90219 # numbers of idx
config.cls_faiss.file_path = '/ your output path /'
config.cls_faiss.db_paths = [] # your db_paths list
config.cls_faiss.np_paths = [] # your np_paths list
