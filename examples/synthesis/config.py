import os
from syszux_config import *
##########text synthesis########
config.text.total_num = 500000
config.text.output_dir = '/gemfield/hostpv2/ocr_synthesis_output/parasite'
config.text.txt_file = '/opt/private/dataset_ocr/synthesis_from_text.txt'
config.text.video_file = '/opt/private/dataset_ocr/synthesis_from_videos/PARASITE.mp4'
config.text.images_dir = '/opt/private/dataset_ocr/synthesis_from_images'
config.text.sample_rate = 1
#config.text.fonts_dir = '/home/lihang/dataset/Ocr1/video/font/'
config.text.fonts_dir = '/gemfield/hostpv2/SYSZUXfont/font/'
config.text.is_border = True
