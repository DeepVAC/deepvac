import os
from deepvac import config, AttrDict
##########text synthesis########
config.text = AttrDict()
config.text.total_num = 189956
config.text.output_dir = '/gemfield/hostpv/ocr_synthesis_output/para'
config.text.txt_file = '/opt/private/dataset_ocr/synthesis_from_text.txt'
config.text.video_file = '/opt/private/dataset_ocr/synthesis_from_videos/PARASITE.mp4'
config.text.images_dir = '/opt/private/dataset_ocr/synthesis_from_images'
config.text.sample_rate = 1
#config.text.fonts_dir = '/home/lihang/dataset/Ocr1/video/font/'
config.text.fonts_dir = '/home/lihang/dataset/Ocr1/video/font/'
config.text.is_border = 0.5
