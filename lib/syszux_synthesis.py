from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import os

class SynthesisBase(object):
    def __init__(self, deepvac_config):
        self.deepvac_config = deepvac_config

    def auditConfig(self):
        self.total_num = self.deepvac_config.total_num
        self.output_path = self.deepvac_config.output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def __call__():
        pass

    def dumpImgToPath(self, file_name, img):
        output_file_name = os.path.join(self.output_path, file_name)
        try:
            cv2.imwrite(output_file_name, img)
        except:
            target_dir = os.path.dirname(os.path.abspath(output_file_name))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            cv2.imwrite(output_file_name, img)

class SynthesisText(SynthesisBase):
    def __init__(self, deepvac_config):
        super(SynthesisText, self).__init__(deepvac_config)
        self.deepvac_config = deepvac_config
        self.lex = []
        self.pil_img = None
        self.draw = None
        self.auditConfig()

    def auditConfig(self):
        super(SynthesisText, self).auditConfig()
        self.txt_file = self.deepvac_config.txt_file
        assert os.path.isfile(self.txt_file), "txt file {} not exist.".format(self.txt_file)

        with open(self.txt_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                self.lex.append(line)

        self.fonts_dir = '/home/lihang/dataset/Ocr1/video/font/'
        if os.path.exists(self.fonts_dir)==False:
            raise Exception("Dir {} not found!".format(self.fonts_dir))
        self.fonts = os.listdir(self.fonts_dir)
        self.fonts_len = len(self.fonts)
        if self.fonts_len == 0:
            raise Exception("No font was found in {}!".format(self.fonts_dir))
        self.font_size = 50
        self.max_font = 60
        self.min_font = 25
        self.crop_scale = 4

        self.fg_color = [(10,10,10),(200,10,10),(10,10,200),(200,200,10),(255,255,255)]
        self.fg_color_len = len(self.fg_color)

    def buildScene(self,i):
        raise Exception("Not implemented!")
    
    def buildTextWithScene(self, i):
        raise Exception("Not implemented!")

    def dumpTextImg(self,i):
        raise Exception("Not implemented!")

    def text_border(self, x, y, font, shadowcolor, fillcolor,text):
        shadowcolor = 'black' if fillcolor==(255,255,255) else 'white'
        for i in [x-1,x+1,x]:
            for j in [y-1,y+1,y]:
                self.draw.text((i, j), text, font=font, fill=shadowcolor)
        self.draw.text((x,y),text,fillcolor,font=font)

    def __call__(self):
        for i in range(self.total_num):
            self.buildScene(i)
            self.buildTextWithScene(i)
            self.dumpTextImg(i)

class SynthesisTextPure(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextPure, self).__init__(deepvac_config)
    
    def auditConfig(self):
        super(SynthesisTextPure, self).auditConfig()
        self.bg_color = [(255,255,255),(10,10,200),(200,10,10),(10,10,200),(10,10,10)]
        self.bg_color_len = len(self.bg_color)
        self.scene_hw = (1080, 1920)
        self.font_offset = (1000,800)
        self.is_border = self.deepvac_config.is_border

    def buildScene(self, i):
        r_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][0]
        g_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][1]
        b_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][2]
        frame = cv2.merge((b_channel, g_channel, r_channel))
        self.pil_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)

    def buildTextWithScene(self, i):
        self.font_size = np.random.randint(self.min_font,self.max_font+1)
        font = ImageFont.truetype(os.path.join(self.fonts_dir,self.fonts[i%self.fonts_len]),self.font_size,encoding='utf-8')
        s = self.lex[i]
        fillcolor = self.fg_color[i%self.fg_color_len]
        if self.is_border:
            self.text_border(self.font_offset[0],self.font_offset[1],font,"white",fillcolor,s)
        else:
            self.draw.text(self.font_offset,s,fillcolor,font=font)
    
    def dumpTextImg(self, i):
        crop_offset = int(self.font_size / self.crop_scale)
        crop_list = [np.random.randint(-crop_offset, crop_offset+1) for x in range(3)]
        cv2_text_im = cv2.cvtColor(np.array(self.pil_img),cv2.COLOR_RGB2BGR)
        img_crop = cv2_text_im[self.font_offset[1]+crop_list[0]:self.font_offset[1]+self.font_size, self.font_offset[0]+crop_list[1]:self.font_offset[0]+self.font_size*len(self.lex[i])+crop_list[2]]
        self.dumpImgToPath('pure_{}.jpg'.format(str(i).zfill(6)),img_crop)

class SynthesisTextFromVideo(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextFromVideo, self).__init__(deepvac_config)

    def auditConfig(self):
        super(SynthesisTextFromVideo, self).auditConfig()
        self.video_file = self.deepvac_config.video_file

        self.video_capture = cv2.VideoCapture(self.video_file)
        self.frames_num = self.video_capture.get(7)
        assert self.frames_num > 10, "invalid video file {}".format(self.video_file)
        self.sample_rate = self.deepvac_config.sample_rate
        self.font_offset = (1000,800)
        self.is_border = self.deepvac_config.is_border

    def buildScene(self, i):
        for _ in range(self.sample_rate):
            success,frame = self.video_capture.read()

            if not success:
                return 
            
        self.pil_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)

    def buildTextWithScene(self, i):
        self.font_size = np.random.randint(self.min_font,self.max_font+1)
        font = ImageFont.truetype(os.path.join(self.fonts_dir,self.fonts[i%self.fonts_len]), self.font_size,encoding='utf-8')
        s = self.lex[i]
        fillcolor = self.fg_color[i%self.fg_color_len]
        if self.is_border:
            self.text_border(self.font_offset[0],self.font_offset[1],font,"white",fillcolor,s)
        else:
            self.draw.text(self.font_offset,s,fillcolor,font=font)

    def dumpTextImg(self, i):
        crop_offset = int(self.font_size / self.crop_scale)
        crop_list = [np.random.randint(-crop_offset, crop_offset+1) for x in range(3)]
        cv2_text_im = cv2.cvtColor(np.array(self.pil_img),cv2.COLOR_RGB2BGR)
        img_crop = cv2_text_im[self.font_offset[1]+crop_list[0]:self.font_offset[1]+self.font_size, self.font_offset[0]+crop_list[1]:self.font_offset[0]+self.font_size*len(self.lex[i])+crop_list[2]]
        self.dumpImgToPath('scene_{}.jpg'.format(str(i).zfill(6)),img_crop)


