import os
from syszux_config import *
##########classifier report########
config.cls = AttrDict()
config.cls.feature_name = 'asia_emor_fix_merged'
config.cls.file_path = './test_report.txt'
config.cls.cls_num = 90219
config.cls.db_paths = ['/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_1.feature', '/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_2.feature', '/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_3.feature', '/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_4.feature']
config.cls.map_locs = [{'cuda:0':'cuda:1'}, {'cuda:0':'cuda:1'}, {'cuda:1':'cuda:0'}, {'cuda:1':'cuda:0'}]
config.cls.np_paths = ['/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_1.feature.npz', '/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_2.feature.npz', '/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_3.feature.npz', '/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_4.feature.npz']
