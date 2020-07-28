import os
from syszux_config import *
##########text synthesis########
config.text.total_num = 100
config.text.output_dir = '/gemfield/hostpv/dataset/deepvac-ocr/output_dir/'
config.text.txt_file = '/opt/private/dataset_ocr/synthesis_from_text.txt'
config.text.video_file = '/opt/private/dataset_ocr/synthesis_from_videos/PARASITE.mp4'
config.text.images_dir = '/opt/private/dataset_ocr/synthesis_from_images'
config.text.sample_rate = 10
config.text.fonts_dir = '/home/lihang/dataset/Ocr1/video/font/'
#config.text.font_dir = '/gemfield/hostpv/dataset/deepvac-ocr/font/'
config.text.is_border = True
