from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import os.path

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
        super(SynthesisText, self).__init__()
        self.deepvac_config = deepvac_config
        self.lex = []
        self.pil_img = None
        self.draw = None
        self.auditConfig()

    def auditConfig(self):
        super(SynthesisText, self).auditConfig(self)
        self.txt_file = self.deepvac_config.txt_file
        assert os.path.isfile(self.txt_file), "txt file {} not exist.".format(self.txt_file)

        with open(self.txt_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                self.lex.append(line)

        self.fonts = ['锐字云字库锐宋粗体GBK.TTF','song.TTF','hei.ttf']
        self.fonts_len = len(self.fonts)
        self.font_size = 50

        self.fg_color = [(10,10,10),(200,10,10),(10,10,200),(200,200,10),(255,255,255)]
        self.fg_color_len = len(self.fg_color)

    def buildScene(self,i):
        raise Exception("Not implemented!")
    
    def buildTextWithScene(self, i):
        raise Exception("Not implemented!")

    def dumpTextImg(self,i):
        raise Exception("Not implemented!")

    def __call__():
        for i in range(self.total_num):
            self.buildScene(i)
            self.buildTextWithScene(i)
            self.dumpTextImg(i)

class SynthesisTextPure(SynthesisText):
    def __init__(self, deepvac_config):
        super(SysthesisTextPure, self).__init__(deepvac_config)
    
    def auditConfig(self):
        super(SysthesisTextPure, self).auditConfig(self)
        self.bg_color = [(255,255,255),(10,10,200),(200,10,10),(10,10,200),(10,10,10)]
        self.bg_color_len = len(self.bg_color)
        self.scene_hw = (1080, 1920)
        self.font_offset = (1000,800)

    def buildScene(self, i):
        r_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][0]
        g_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][1]
        b_channel = np.ones(self.scene_hw, dtype=np.uint8) * self.bg_color[i%self.bg_color_len][2]
        frame = cv2.merge((b_channel, g_channel, r_channel))
        self.pil_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)

    def buildTextWithScene(self, i):
        font = ImageFont.truetype(self.fonts[i%self.fonts_len], self.font_size,encoding='utf-8')
        s = self.lex[i]
        self.draw.text(self.font_offset,s,self.fg_color[i%self.fg_color_len],font=font)

    def dumpTextImg(self, i):
        cv2_text_im = cv2.cvtColor(np.array(self.pil_img),cv2.COLOR_RGB2BGR)
        img_crop = cv2_text_im[self.font_offset[1]:self.font_offset[1] + self.font_size, self.font_offset[0]:self.font_offset[0] + self.font_size*len(s)]
        self.dumpImgToPath('pure_{}.jpg'.format(str(i).zfill(6)),img_crop)

            
class SynthesisTextFromVideo(SynthesisText):
    def __init__(self, deepvac_config):
        super(SysthesisTextFromVideo, self).__init__(deepvac_config)

    def auditConfig(self):
        super(SysthesisTextPure, self).auditConfig(self)
        self.video_file = self.deepvac_config.video_file

        self.video_capture = cv2.VideoCapture(self.video_file)
        self.frames_num = self.video_capture.get(7)
        assert self.frames_num > 10, "invalid video file {}".format(self.video_file)
        self.sample_rate = self.deepvac_config.sample_rate

    def buildScene(self, i):
        read_num = 0
        while True:
            for _ in range(self.sample_rate):
                success,frame = self.video_capture.read()
                read_num += 1

                if read_num >= self.frames_num:
                    return

                if not success:
                    continue
            
            self.pil_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            self.draw = ImageDraw.Draw(pil_img)
            yield

    def buildTextWithScene(self, i):
        pass

    def dumpTextImg(self, i):
        pass






