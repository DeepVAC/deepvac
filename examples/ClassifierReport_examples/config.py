import os
from deepvac.syszux_config import *
##########classifier report########
config.cls = AttrDict()
config.cls.class_num = 90219  # numbers of idx
config.cls.log_every = 10000 # log
config.cls.db_path_list = [] # your db_paths list
config.cls.np_path_list = []  # your np_paths list
config.cls.map_locs = [{'cuda:0':'cuda:1'}] # your map_locs config
config.cls.file_path = './test_report.txt'  # result file path
##########classifier_faiss report########
config.cls_faiss = AttrDict()
config.cls_faiss.class_num = 90219 # numbers of idx
config.cls_faiss.log_every = 10000 # log
config.cls_faiss.db_path_list = [] # your db_paths list
config.cls_faiss.np_path_list = [] # your np_paths list
config.cls_faiss.file_path = './test_report.txt' # result file path
##########classifier_faiss_pth report########
config.cls_faiss_pth = AttrDict()
config.cls_faiss_pth.class_num = 90219  # numbers of idx
config.cls_faiss_pth.log_every = 10000  # log
config.cls_faiss_pth.db_path_list = []   # your db_paths list
config.cls_faiss_pth.np_path_list = []   # your np_paths list
config.cls_faiss_pth.map_locs = [{'cuda:0':'cuda:1'}] # your map_locs config
config.cls_faiss_pth.file_path = './test_report.txt' # result file path
