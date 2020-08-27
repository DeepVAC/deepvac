from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import os
import random

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

    def dumpTextImg(self,i):
        crop_offset = int(self.font_size / self.crop_scale)
        crop_list = [np.random.randint(-crop_offset, crop_offset+1) for x in range(3)]
        cv2_text_im = cv2.cvtColor(np.array(self.pil_img),cv2.COLOR_RGB2BGR)
        img_crop = cv2_text_im[self.font_offset[1]+crop_list[0]:self.font_offset[1]+self.font_size+10, self.font_offset[0]+crop_list[1]:self.font_offset[0]+self.font_size*len(self.lex[i%self.lex_len])+crop_list[2]]
        image_name = '{}_{}.jpg'.format(self.dump_prefix,str(i).zfill(6))
        self.dumpImgToPath(image_name,img_crop)
        self.fw.write(image_name+' '+self.lex[i%self.lex_len]+'\n')

    def __call__(self):
        for i in range(self.total_num):
            self.buildScene(i)
            self.buildTextWithScene(i)
            self.dumpTextImg(i)
            if i%5000==0:
                print('{}/{}'.format(i,self.total_num))

class SynthesisTextPure(SynthesisText):
    def __init__(self, deepvac_config):
        super(SynthesisTextPure, self).__init__(deepvac_config)

    def __exit__(self):
        self.fw.close()
    
    def auditConfig(self):
        super(SynthesisTextPure, self).auditConfig()
        self.bg_color = [(255,255,255),(10,10,200),(200,10,10),(10,10,200),(10,10,10)]
        self.bg_color_len = len(self.bg_color)
        self.scene_hw = (1080, 1920)
        self.font_offset = (1000,800)
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
        self.font_size = np.random.randint(self.min_font,self.max_font+1)
        font = ImageFont.truetype(os.path.join(self.fonts_dir,self.fonts[i%self.fonts_len]),self.font_size,encoding='utf-8')
        s = self.lex[i%self.lex_len]
        fillcolor = self.fg_color[i%self.fg_color_len]
        if self.is_border:
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
        self.font_offset = (int(self.max_font/self.crop_scale),int(self.frame_height/3-self.max_font))
        self.is_border = self.conf.is_border
        self.dump_prefix = 'scene'
        self.fw = open(os.path.join(self.conf.output_dir,'video.txt'),'w')

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
        s = self.lex[i%self.lex_len]
        fillcolor = self.fg_color[i%self.fg_color_len]
        if self.is_border:
            self.text_border(self.font_offset[0],self.font_offset[1],font,"white",fillcolor,s)
        else:
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
        if self.images_num<self.total_num:
            raise Exception("Total_num {} exceeds the image numbers {}, build exit!".format(self.total_num, self.images_num))
        self.scene_hw = (1080,1920)
        self.font_offset = (1000, 800)
        self.is_border = self.conf.is_border
        self.dump_prefix = 'image'
        self.fw = open(os.path.join(self.conf.output_dir,'image.txt'),'w')

    def buildScene(self, i):
        image = cv2.imread(os.path.join(self.images_dir, self.images[i]))
        image = cv2.resize(image,(self.scene_hw[1],self.scene_hw[0]))
        self.pil_img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)

    def buildTextWithScene(self, i):
        self.font_size = np.random.randint(self.min_font,self.max_font+1)
        font = ImageFont.truetype(os.path.join(self.fonts_dir,self.fonts[i%self.fonts_len]), self.font_size,encoding='utf-8')
        s = self.lex[i%self.lex_len]
        fillcolor = self.fg_color[i%self.fg_color_len]
        if self.is_border:
            self.text_border(self.font_offset[0],self.font_offset[1],font,"white",fillcolor,s)
        else:
            self.draw.text(self.font_offset,s,fillcolor,font=font)
