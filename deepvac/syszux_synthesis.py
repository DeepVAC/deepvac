from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import os
import random
from .syszux_log import LOG
from .syszux_helper import Haishoku
try:
    from fontTools.ttLib import TTFont
except:
    LOG.logE('no fonttools, pip install fonttools please', exit=True)

class SynthesisBase(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.auditConfig()

    def auditConfig(self):
        self.total_num = self.conf.total_num
        self.output_dir = self.conf.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __call__():
        pass

    def dumpImgToPath(self, file_name, img):
        output_file_name = os.path.join(self.output_dir, file_name)
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
        self.i_just_want_font = None

    def getFontTables(self, font_path):
        return TTFont(font_path)

    def hasGlyph(self, font, glyph):
        for table in font['cmap'].tables:
            if ord(glyph) in table.cmap.keys():
                return True
        return False

    def auditConfig(self):
        super(SynthesisText, self).auditConfig()
        self.lex = []
        self.pil_img = None
        self.draw = None
        self.txt_file = self.conf.txt_file
        assert os.path.isfile(self.txt_file), "txt file {} not exist.".format(self.txt_file)

        with open(self.txt_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                self.lex.append(line)
        random.shuffle(self.lex)
        self.lex_len = len(self.lex)

        self.fonts_dir = self.conf.fonts_dir
        if not os.path.exists(self.fonts_dir):
            raise Exception("Dir {} not found!".format(self.fonts_dir))
        self.font_file_list = os.listdir(self.fonts_dir)
        print(self.font_file_list)
        self.font_num = len(self.font_file_list)
        if self.font_num == 0:
            raise Exception("No font was found in {}!".format(self.fonts_dir))
        self.current_font_size = 50
        self.max_font = 60
        self.min_font = 25
        self.crop_scale = 4
        self.scene_hw = (1080,1920)
        self.gb18030_font_file_list = []

        for i, font_file in enumerate(self.font_file_list):
            LOG.logI("found font: {}:{}".format(i, font_file))
            if font_file.startswith('gb18030'):
                LOG.logI("And this is a gb18030 font!")
                self.gb18030_font_file_list.append(font_file)

        self.runtime_fonts = dict()
        self.runtime_gb18030_fonts = dict()
        for font_size in range (self.min_font, self.max_font + 1):
            self.runtime_fonts[font_size] = []
            for font_file in self.font_file_list:
                font = ImageFont.truetype(os.path.join(self.fonts_dir,font_file), font_size, encoding='utf-8')
                self.runtime_fonts[font_size].append(font)

        for font_size in range (self.min_font, self.max_font + 1):
            self.runtime_gb18030_fonts[font_size] = []
            for font_file in self.gb18030_font_file_list:
                font = ImageFont.truetype(os.path.join(self.fonts_dir,font_file), font_size, encoding='utf-8')
                self.runtime_gb18030_fonts[font_size].append(font)

        self.support_fonts4char = {}
        with open(self.conf.chars, 'r') as f:
            line = f.readlines()[0][:-1]
            for c in line:
                self.support_fonts4char[c]= []
            for idx, font_file in enumerate(self.font_file_list):
                font_table = self.getFontTables(os.path.join(self.fonts_dir,font_file))
                for c in line:
                    if self.hasGlyph(font_table, c):
                        self.support_fonts4char[c].append(idx)
                    elif font_file in self.gb18030_font_file_list:
                        LOG.logE('{} not supported in current font! Are you sure {} is a gb18030 font?'.format(c, font_file))

        self.fg_color = [(10,10,10),(200,10,10),(10,10,200),(200,200,10),(255,255,255)]
        self.fg_color_len = len(self.fg_color)

        self.distance = 100  # The min distance of fg_color and bg_color 

    def buildScene(self,i):
        raise Exception("Not implemented!")
    
    def buildTextWithScene(self, i):
        raise Exception("Not implemented!")

    def dumpTextImg(self,i):
        raise Exception("Not implemented!")

    def pickFgColor(self, i, s):
        left = self.font_offset[0]
        up = self.font_offset[1] 
        right = self.font_offset[0] + len(s)*self.max_font
        below = self.font_offset[1] + self.max_font
        dominant = Haishoku.getDominant(self.pil_img.crop((left,up,right,below)))

        k = i % len(self.fg_color)
        fg_lst = self.fg_color[k:] + self.fg_color[:k]
        max_dis = 0
        for fg in fg_lst:
            distance = abs(dominant[0]-fg[0]) + abs(dominant[1]-fg[1]) + abs(dominant[2]-fg[2])
            #print(self.images[i%self.images_num], dominant, fg, distance)
            if distance > self.distance:
                return fg
            if distance > max_dis:
               max_dis_fg = fg
               max_dis = distance
        LOG.logI("No fg_color is suitable for image {} !!!".format(i))
        return max_dis_fg

    def setCurrentFontSizeAndGetFont(self, i):
        s = self.lex[i%self.lex_len]
        self.current_font_size = np.random.randint(self.min_font,self.max_font+1)
        if self.i_just_want_font is not None:
            return s, self.runtime_fonts[self.current_font_size][self.i_just_want_font % self.font_num]
        
        font_idx = i%self.font_num
        font = self.runtime_fonts[self.current_font_size][i%self.font_num]
        print(s)
        for c in s:
            if font_idx in self.support_fonts4char[c]:
                continue
            font = self.runtime_gb18030_fonts[self.current_font_size][np.random.randint(0,len(self.runtime_gb18030_fonts[self.current_font_size]))]
            print(font)
            break

        return s, font

    def text_border(self, x, y, font, shadowcolor, fillcolor,text):
        shadowcolor = 'black' if fillcolor==(255,255,255) else 'white'
        for i in [x-1,x+1,x]:
            for j in [y-1,y+1,y]:
                self.draw.text((i, j), text, font=font, fill=shadowcolor)
        self.draw.text((x,y),text,fillcolor,font=font)

    def dumpTextImg(self,i):
        crop_offset = int(self.current_font_size / self.crop_scale)
        #crop_list = [np.random.randint(-crop_offset, crop_offset+1) for _ in range(3)]
        cv2_text_im = cv2.cvtColor(np.array(self.pil_img),cv2.COLOR_RGB2BGR)
        img_crop = cv2_text_im[self.font_offset[1]+np.random.randint(-crop_offset, crop_offset/2):self.font_offset[1]+self.current_font_size + np.random.randint(crop_offset/2, crop_offset * 1.5),
                self.font_offset[0]+np.random.randint(-crop_offset, crop_offset+1):self.font_offset[0]+self.current_font_size*len(self.lex[i%self.lex_len])+np.random.randint(-crop_offset, crop_offset+1)]
        image_name = '{}_{}.jpg'.format(self.dump_prefix,str(i).zfill(6))
        self.dumpImgToPath(image_name,img_crop)
        self.fw.write(image_name+' '+self.lex[i%self.lex_len]+'\n')

    def __call__(self):
        for i in range(self.total_num):
            self.buildScene(i)
            self.buildTextWithScene(i)
            self.dumpTextImg(i)
            if i%5000==0:
                LOG.logI('{}/{}'.format(i,self.total_num))

class SynthesisTextPure(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextPure, self).__init__(deepvac_config)

    def __exit__(self):
        self.fw.close()
    
    def auditConfig(self):
        super(SynthesisTextPure, self).auditConfig()
        self.bg_color = [(255,255,255),(10,10,200),(200,10,10),(10,10,200),(10,10,10)]
        self.bg_color_len = len(self.bg_color)
        self.font_offset = (self.max_font,800)
        self.is_border = self.conf.is_border
        self.fw = open(os.path.join(self.conf.output_dir,'pure.txt'),'w')

    def buildScene(self, i):
        r_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][0]
        g_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][1]
        b_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][2]
        frame = cv2.merge((b_channel, g_channel, r_channel))
        self.pil_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)
        self.dump_prefix = 'pure'

    def buildTextWithScene(self, i):
        s, font = self.setCurrentFontSizeAndGetFont(i)
        fillcolor = self.fg_color[i%self.fg_color_len]
        if np.random.rand() < self.is_border:
            self.text_border(self.font_offset[0],self.font_offset[1],font,"white",fillcolor,s)
        else:
            self.draw.text(self.font_offset,s,fillcolor,font=font)

class SynthesisTextFromVideo(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextFromVideo, self).__init__(deepvac_config)

    def __exit__(self):
        self.fw.close()

    def auditConfig(self):
        super(SynthesisTextFromVideo, self).auditConfig()
        self.video_file = self.conf.video_file

        self.video_capture = cv2.VideoCapture(self.video_file)
        self.frames_num = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        assert self.frames_num > 10, "invalid video file {}".format(self.video_file)
        self.sample_rate = self.conf.sample_rate
        if self.frames_num/self.sample_rate<self.total_num:
            raise Exception("Total_num {} exceeds frame_nums({})/sample_rate({}), build exit!".format(self.total_num,int(self.frames_num),self.sample_rate))
        self.frame_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        assert self.frame_height > 4 * self.max_font, "video height must exceed {} pixels".format(4*self.max_font)
        self.frame_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.resize_ratio = None

        if self.frame_width < self.scene_hw[1]:
            self.resize_ratio = self.scene_hw[1] / self.frame_width
            self.frame_height = int(self.frame_height * self.resize_ratio)

        self.font_offset = (int(self.max_font/self.crop_scale),int(self.frame_height/3-self.max_font))
        self.is_border = self.conf.is_border
        self.dump_prefix = 'scene'
        self.fw = open(os.path.join(self.conf.output_dir,'video.txt'),'w')

    def buildScene(self, i):
        for _ in range(self.sample_rate):
            success,frame = self.video_capture.read()

            if not success:
                return
        if self.resize_ratio is not None:
            frame = cv2.resize(frame,(self.scene_hw[1], self.frame_height))

        self.pil_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)

    def buildTextWithScene(self, i):
        s, font = self.setCurrentFontSizeAndGetFont(i)
        if np.random.rand() < self.is_border:
            fillcolor = self.fg_color[i%self.fg_color_len]
            self.text_border(self.font_offset[0],self.font_offset[1],font,"white",fillcolor,s)
        else:
            fillcolor = self.pickFgColor(i, s)
            self.draw.text(self.font_offset,s,fillcolor,font=font)

class SynthesisTextFromImage(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextFromImage, self).__init__(deepvac_config)

    def __exit__(self):
        self.fw.close()

    def auditConfig(self):
        super(SynthesisTextFromImage, self).auditConfig()
        self.images_dir = self.conf.images_dir
        if not os.path.exists(self.images_dir):
            raise Exception("Dir {}not found!".format(self.images_dir))
        self.images = os.listdir(self.images_dir)
        self.images_num = len(self.images)
        if self.images_num==0:
            raise Exception("No image was found in {}!".format(self.images))
        self.font_offset = (int(self.max_font/self.crop_scale),int(self.scene_hw[0]/3-self.max_font))
        self.is_border = self.conf.is_border
        self.dump_prefix = 'image'
        self.fw = open(os.path.join(self.conf.output_dir,'image.txt'),'w')

    def buildScene(self, i):
        image = cv2.imread(os.path.join(self.images_dir, self.images[i%self.images_num]))
        image = cv2.resize(image,(self.scene_hw[1],self.scene_hw[0]))
        self.pil_img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)

    def buildTextWithScene(self, i):
        s, font = self.setCurrentFontSizeAndGetFont(i)
        if np.random.rand() < self.is_border:
            fillcolor = self.fg_color[i%self.fg_color_len]
            self.text_border(self.font_offset[0],self.font_offset[1],font,"white",fillcolor,s)
        else:
            fillcolor = self.pickFgColor(i, s)
            self.draw.text(self.font_offset,s,fillcolor,font=font)
