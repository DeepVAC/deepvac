# -*- coding:utf-8 -*-
# Author: RubanSeven
import math
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from syszux_helper import WarpMLS

class AugBase(object):
    def __init__(self, deepvac_config):
        self.auditConfig()

    def auditConfig(self):
        raise Exception("Not implemented!")

    def __call__(self,img):
        raise Exception("Not implemented!")

    def pillow2cv(self, pillow_img, is_rgb2bgr=True):
        cv_image = np.array(pillow_img)
        if is_bgr2rgb:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image

    def cv2pillow(self, cv_img, is_bgr2rgb=True):
        if is_bgr2rgb:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv_img)

# 图像添加随机斑点
class SpeckleAug(AugBase):
    def __init__(self, deepvac_config):
        super(SpeckleAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.severity = np.random.uniform(0, 0.6*255)

    def __call__(self, img):
        blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * self.severity, 1)
        img_speck = (img + blur)
        img_speck[img_speck > 255] = 255
        img_speck[img_speck <= 0] = 0
        return img_speck

# 仿射变换
class AffineAug(AugBase):
    def __init__(self, deepvac_config):
        super(AffineAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # 空白填充色
        self.borderValue = (255,255,255)
        # x方向和y方向的伸缩率
        self.shear_x = 30
        self.shear_y = 1

    def __call__(self, img):
        rows,cols=img.shape[:2]
        shear_x = float(np.random.randint(-self.shear_x,self.shear_x+1))/100
        shear_y = float(np.random.randint(-self.shear_y,self.shear_y+1))/100
        M = np.float32([[1.0, shear_x ,0.0],[shear_y, 1.0,0.0]])
        img_affine = cv2.warpAffine(img,M,(cols,rows),borderValue=self.borderValue)
        return img_affine

# 透视变换
class PerspectAug(AugBase):
    def __init__(self, deepvac_config):
        super(PerspectAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # b空白填充色
        self.borderValue = (255,255,255)
        # 高h方向伸缩范围
        self.sh_lower = 6
        self.sh_upper = 11
        # 宽w方向伸缩范围
        self.sw_lower = 20
        self.sw_upper = 31

    def __call__(self, img):
        h,w=img.shape[:2]
        scale_h = np.random.randint(self.sh_lower ,self.sh_upper)
        scale_w = np.random.randint(self.sw_lower ,self.sw_upper)
        point1 = np.array([[0,0],[w,0],[0,h],[w,h]],dtype = "float32")
        point2_list = [
            np.array([[w/scale_w,0],[(scale_w-1)*w/scale_w,0],[0,h],[w,h]],dtype = "float32"),
            np.array([[0,h/scale_h],[w,0],[0,(scale_h-1)*h/scale_h],[w,h]],dtype = "float32"),
            np.array([[0,0],[w,0],[w/scale_w,h],[(scale_w-1)*w/scale_w,h]],dtype = "float32"),
            np.array([[0,0],[w,h/scale_h],[0,h],[w,(scale_h-1)*h/scale_h]],dtype = "float32")]
        
        pt_idx = np.random.randint(0,4)
        M = cv2.getPerspectiveTransform(point1,point2_list[pt_idx])
        img_perspect = cv2.warpPerspective(img,M,(w,h),borderValue=self.borderValue)
        return img_perspect

# 高斯模糊
class GaussianAug(AugBase):
    def __init__(self, deepvac_config):
        super(GaussianAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.ks = 5

    def __call__(self, img):
        img_gaussian = cv2.GaussianBlur(img,(self.ks, self.ks), 0)
        return img_gaussian

# 添加横线增强
class HorlineAug(AugBase):
    def __init__(self, deepvac_config):
        super(HorlineAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # 线条间隔
        self.space = 4
        # 线条颜色
        self.color = 0
        # 线宽
        self.thickness = 1

    def __call__(self, img):
        img_horline = img.copy()
        img_h, img_w = img.shape[:2]
        for i in range(0,img_h,self.space):
            cv2.line(img_horline, (0, i), (img_w, i), self.color, self.thickness)
        return img_horline

# 添加竖线增强
class VerlineAug(AugBase):
    def __init__(self, deepvac_config):
        super(VerlineAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # 线条间隔
        self.space = 4
        # 线条颜色
        self.color = 0
        # 线宽
        self.thickness = 1

    def __call__(self, img):
        img_verline = img.copy()
        img_h, img_w = img.shape[:2]
        for i in range(0,img_w,self.space):
            cv2.line(img_verline, (i, 0), (i, img_h), self.color, self.thickness)
        return img_verline

# 左右运动模糊
class LRmotionAug(AugBase):
    def __init__(self, deepvac_config):
        super(LRmotionAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.ks = 5

    def __call__(self,img):
        kernel_motion_blur = np.zeros((self.ks, self.ks))
        kernel_motion_blur[int((self.ks - 1) / 2), :] = np.ones(self.ks)
        kernel_motion_blur = kernel_motion_blur / self.ks
        img_lrmotion = cv2.filter2D(img, -1, kernel_motion_blur)
        return img_lrmotion

# 上下运动模糊
class UDmotionAug(AugBase):
    def __init__(self, deepvac_config):
        super(UDmotionAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.ks = 9

    def __call__(self, img):
        kernel_motion_blur = np.zeros((self.ks, self.ks))
        kernel_motion_blur[:, int((self.ks - 1) / 2)] = np.ones(self.ks)
        kernel_motion_blur = kernel_motion_blur / self.ks
        img_udmotion = cv2.filter2D(img, -1, kernel_motion_blur)
        return img_udmotion

# 添加噪声
class NoisyAug(AugBase):
    def __init__(self, deepvac_config):
        super(NoisyAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.mean = 0
        self.sigma = 1
    
    def __call__(self, img):
        row, col = img.shape[:2]
        gauss = np.random.normal(self.mean, self.sigma, (row, col,3))
        gauss = gauss.reshape(row, col,3)
        noisy = img + gauss
        img_noisy= noisy.astype(np.uint8)
        return img_noisy

# 扭曲变换
class DistortAug(AugBase):
    def __init__(self, deepvac_config):
        super(DistortAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.segment = 4

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        cut = img_w // self.segment
        thresh = cut // 3

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
        dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

        half_thresh = thresh * 0.5
        for cut_idx in np.arange(1, self.segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            np.random.randint(thresh) - half_thresh])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            img_h + np.random.randint(thresh) - half_thresh])
        
        trans = WarpMLS(img, src_pts, dst_pts, img_w, img_h)
        img_distort = trans.generate()
        return img_distort

# 随机伸缩变换
class StretchAug(AugBase):
    def __init__(self, deepvac_config):
        super(StretchAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.segment = 4

    def __call__(self, img):
        img_h, img_w = img.shape[:2]

        cut = img_w // self.segment
        thresh = cut * 4 // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, self.segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        trans = WarpMLS(img, src_pts, dst_pts, img_w, img_h)
        img_stretch = trans.generate()
        return img_stretch

# 透视变换
class PerspectiveAug(AugBase):
    def __init__(self, deepvac_config):
        super(PerspectiveAug,self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        img_h, img_w = img.shape[:2]

        thresh = img_h // 2
        
        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        trans = WarpMLS(img, src_pts, dst_pts, img_w, img_h)
        img_perspective = trans.generate()
        return img_perspective
