# -*- coding:utf-8 -*-
# Author: RubanSeven
import math
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
from scipy import ndimage
from .syszux_helper import WarpMLS, apply_perspective_transform, Remaper, Liner, apply_emboss, reverse_img

class AugBase(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.auditConfig()

    def auditConfig(self):
        raise Exception("Not implemented!")

    def __call__(self,img):
        raise Exception("Not implemented!")

    @staticmethod
    def pillow2cv(pillow_img, is_rgb2bgr=True):
        cv_image = np.array(pillow_img)
        if is_bgr2rgb:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image

    @staticmethod
    def cv2pillow(cv_img, is_bgr2rgb=True):
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
        self.ks = [9,11,13,15,17]

    def __call__(self, img):
        ks = self.ks[np.random.randint(0,len(self.ks))]
        img_gaussian = cv2.GaussianBlur(img,(ks, ks), 0)
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
        self.ks = [3,5,7,9]

    def __call__(self,img):
        ks = self.ks[np.random.randint(0,len(self.ks))]
        kernel_motion_blur = np.zeros((ks, ks))
        kernel_motion_blur[int((ks - 1) / 2), :] = np.ones(ks)
        kernel_motion_blur = kernel_motion_blur / ks
        img_lrmotion = cv2.filter2D(img, -1, kernel_motion_blur)
        return img_lrmotion

# 上下运动模糊
class UDmotionAug(AugBase):
    def __init__(self, deepvac_config):
        super(UDmotionAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.ks = [3,5,7,9]

    def __call__(self, img):
        ks = self.ks[np.random.randint(0,len(self.ks))]
        kernel_motion_blur = np.zeros((ks, ks))
        kernel_motion_blur[:, int((ks - 1) / 2)] = np.ones(ks)
        kernel_motion_blur = kernel_motion_blur / ks
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
        if thresh == 0:
            return img

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
        if thresh==0:
            return img

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
        if thresh==0:
            return img

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

# 运动模糊
class MotionAug(AugBase):
    def __init__(self, deepvac_config):
        super(MotionAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.degree = 18
        self.angle = 45

    def __call__(self, img):
        m = cv2.getRotationMatrix2D((self.degree / 2, self.degree / 2), self.angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (self.degree, self.degree))
        motion_blur_kernel = motion_blur_kernel / self.degree
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)

        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        return blurred

# 降低图片亮度
class DarkAug(AugBase):
    def __init__(self, deepvac_config):
        super(DarkAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.gamma = 3

    def __call__(self, img):
        is_gray = img.ndim == 2 or img.shape[1] == 1
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        illum = np.power(illum, self.gamma)
        v = illum * 255.
        v[v > 255] = 255
        v[v < 0] = 0
        hsv[..., 2] = v.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

# 降低图片半边亮度
class HalfDarkAug(AugBase):
    def __init__(self, deepvac_config):
        super(HalfDarkAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.gamma = 1.5

    def __call__(self, img):
        h, w, _ = img.shape
        is_gray = img.ndim == 2 or img.shape[1] == 1
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        illum[:, w//2:] = np.power(illum[:, w//2:], self.gamma)
        v = illum * 255
        v[v > 255] = 255
        v[v < 0] = 0
        hsv[..., 2] = v.astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

# 模拟IPC场景增强
class IPCFaceAug(AugBase):
    def __init__(self, deepvac_config):
        super(IPCFaceAug, self).__init__(deepvac_config)
        self.deepvac_config = deepvac_config

    def auditConfig(self):
        pass

    def __call__(self, img):
        half_dark = HalfDarkAug(self.deepvac_config)
        half_dark.auditConfig()
        half_darked = half_dark(img)

        motion = MotionAug(self.deepvac_config)
        motion.auditConfig()
        motioned = motion(half_darked)

        return motioned

# 随机crop框降低亮度
class RandomCropDarkAug(AugBase):
    def __init__(self, deepvac_config):
        super(RandomCropDarkAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.gamma = 1.2

    def __call__(self, img):
        height, width, _ = img.shape
        w = np.random.uniform(0.3 * width, width)
        h = np.random.uniform(0.3 * height, height)
        if h / w < 0.5 or h / w > 2:
            return img
        left = np.random.uniform(width - w)
        top = np.random.uniform(height - h)

        rect = np.array([int(left), int(top), int(left+w), int(top+h)])
        current_img = img[rect[1]:rect[3], rect[0]:rect[2], :]
        hsv = cv2.cvtColor(current_img, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        illum = np.power(illum, self.gamma)
        v = illum * 255.
        v[v > 255] = 255
        v[v < 0] = 0
        hsv[..., 2] = v.astype(np.uint8)
        dark_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for x in range(rect[1], rect[3]):
            for y in range (rect[0], rect[2]):
                img[x, y, 0] = dark_img[x-rect[1], y-rect[0], 0]
                img[x, y, 1] = dark_img[x-rect[1], y-rect[0], 1]
                img[x, y, 2] = dark_img[x-rect[1], y-rect[0], 2]

        print('rect:', rect)
        return img

# 随机颜色扰动
class RandomColorJitterAug(AugBase):
    def __init__(self, deepvac_config):
        super(RandomColorJitterAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        random.seed()
        change_image = random.randint(0, 1)
        if not change_image:
            return img
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if random.randint(0, 1):
            img = ImageEnhance.Color(img).enhance(np.random.uniform(0.8, 1.3))
        if random.randint(0, 1):
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.6, 1.5))
        if random.randint(0, 1):
            img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.5, 1.8))

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class MosaicAug(AugBase):
    def __init__(self, deepvac_config):
        super(MosaicAug, self).__init__(deepvac_config)
        self.neighbor = 2

    def auditConfig(self):
        pass

    def __call__(self, img):
        neighbor = self.neighbor
        h, w = img.shape[0], img.shape[1]
        for i in range(0, h - neighbor -1, neighbor):
            for j in range(0, w - neighbor - 1, neighbor):
                rect = [j, i, neighbor, neighbor]
                color = img[i][j].tolist()
                left_up = (rect[0], rect[1])
                right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)
                cv2.rectangle(img, left_up, right_down, color, -1)
        return img


# 随机旋转（针对于人脸关键点任务）
class RandomRotateFacialKpListAug(AugBase):
    def __init__(self, deepvac_config):
        super(RandomRotateFacialKpListAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    # img_list: include 2 element
    # img_list[0] -> img
    # img_list[1] -> keypoint info (scale to 0-1)
    def __call__(self, img_list):
        assert isinstance(img_list, list), 'input must be a list'
        assert len(img_list) == 2, 'img_list length must be two'

        img = img_list[0]
        landmarks = img_list[1]
        angle = random.choice([-13, -12, -11, -10, -9, -8, -7, -6, -5, 0, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10, 11, 12, 13])
        h, w, _ = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        dest_img = cv2.warpAffine(img, M, (w, h))
        dest_landmarks = []
        for i in range(len(landmarks)):
            curlandmark_x = landmarks[i][0] * w
            curlandmark_y = landmarks[i][1] * h
            dst_x = curlandmark_x * M[0][0] + curlandmark_y * M[0][1] + M[0][2]
            dst_y = curlandmark_x * M[1][0] + curlandmark_y * M[1][1] + M[1][2]
            dest_landmarks.append([dst_x / w, dst_y / h])
        return [dest_img, dest_landmarks]

# 随机水平翻转（针对于人脸关键点任务,关键点索引顺序等需要和deepvac人脸关键点索引顺序一致）
class RandomFilpFacialKpListAug(AugBase):
    def __init__(self, deepvac_config):
        super(RandomFilpFacialKpListAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def flipLandmark(self, dest_landmark, src_landmark, sequences):
        for sequence in sequences:
            for i in range(sequence[1], sequence[0] - 1, -1):
                dest_landmark.append(src_landmark[i])
        return dest_landmark

    def copyLandmark(self, dest_landmark, src_landmark, sequences):
        for sequence in sequences:
            for i in range(sequence[0], sequence[1] + 1):
                dest_landmark.append(src_landmark[i])
        return dest_landmark

    # img_list: include 2 element
    # img_list[0] -> img
    # img_list[1] -> keypoint info(scale to 0-1)
    def __call__(self, img_list):
        assert isinstance(img_list, list), 'input must be a list'
        assert len(img_list) == 2, 'img_list length must be two'

        img = img_list[0]
        landmarks = img_list[1]
        h, w, _ = img.shape
        random.seed()

        if random.randint(0, 1) == 0:
            return [img, landmarks]

        dest_img = cv2.flip(img, 1)
        flip_landmarks = []
        for i in range(len(landmarks)):
            curlandmark_x = w - 1 - landmarks[i][0]
            curlandmark_y = landmarks[i][1]
            flip_landmarks.append([curlandmark_x, curlandmark_y])

        dest_landmarks = []

        # 重新排列关键点顺序
        flip_landmarks_list = [[0, 32], [38, 42], [33, 37]]
        dest_landmarks = self.flipLandmark(dest_landmarks, flip_landmarks, flip_landmarks_list)
        dest_landmarks = self.copyLandmark(dest_landmarks, flip_landmarks, [[43, 46]])
        flip_landmarks_list = [[47, 51], [58, 61], [62, 63], [52, 55], [56, 57], [68, 71], [64, 67]]
        dest_landmarks = self.flipLandmark(dest_landmarks, flip_landmarks, flip_landmarks_list)
        dest_landmarks = self.copyLandmark(dest_landmarks, flip_landmarks, [[75, 77], [72, 74]])
        flip_landmarks_list = [[78, 79], [80, 81], [82, 83], [84, 90], [91, 95], [96, 100], [101, 103], [104, 105]]
        dest_landmarks = self.flipLandmark(dest_landmarks, flip_landmarks, flip_landmarks_list)

        return [dest_img, dest_landmarks]

class TextRendererPerspectiveAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererPerspectiveAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.max_x = 10
        self.max_y = 10
        self.max_z = 5

    def __call__(self, img):
        return apply_perspective_transform(img, self.max_x, self.max_y, self.max_z)

class TextRendererCurveAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererCurveAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        h, w = img.shape[:2]
        re_img, text_box_pnts = Remaper().apply(img, [[0,0],[w,0],[w,h],[0,h]])
        return re_img

class TextRendererLineAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererLineAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.offset = 5

    def __call__(self, img):
        h, w = img.shape[:2]
        pos = [[self.offset,self.offset],[w-self.offset,self.offset],[w-self.offset,h-self.offset],[self.offset,h-self.offset]]
        re_img, text_box_pnts = Liner().apply(img, pos)
        return re_img

class TextRendererEmbossAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererEmbossAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        return apply_emboss(img)

class TextRendererReverseAug(AugBase):
    def __init__(self, deepvac_config):
        super(TextRendererReverseAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        return reverse_img(img)


### yolov5 dataset aug
class HSVAug(AugBase):
    def __init__(self, deepvac_config):
        super(HSVAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        hgain = self.conf.hgain
        sgain = self.conf.sgain
        vgain = self.conf.vgain
        # r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        r = np.array([0.99751, 1.3085, 0.60009])
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


class RandomPerspectiveAug(AugBase):
    def __init__(self, deepvac_config):
        super(RandomPerspectiveAug, self).__init__(deepvac_config)

    def auditConfig(self):
        # Perspective
        self.perspective = self.conf.perspective
        # Rotation and Scale
        self.degrees = self.conf.degrees
        self.scale = self.conf.scale
        # Shear
        self.shear = self.conf.shear
        # Translation
        self.translate = self.conf.translate

    def _box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

    def __call__(self, img, target):
        border = self.conf.border
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2
        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)
        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)
        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)
        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        # Transform label coordinates
        n = len(target)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = target[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            # filter candidates
            i = self._box_candidates(box1=target[:, 1:5].T * s, box2=xy.T)
            target = target[i]
            target[:, 1:5] = xy[i]
        return img, target


class FlipAug(AugBase):
    def __init__(self, deepvac_config):
        super(FlipAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.flipud = self.conf.flipud
        self.fliplr = self.conf.fliplr

    def __call__(self, img, target):
        if random.random() < self.flipud:
            img = np.flipud(img)
            if target.size:
                target[:, 2] = 1 - target[:, 2]
        if random.random() < self.fliplr:
            img = np.fliplr(img)
            if target.size:
                target[:, 1] = 1 - target[:, 1]
        return img, target


class CutoutAug(AugBase):
    def __init__(self, deepvac_config):
        super(CutoutAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def _bbox_ioa(self, box1, box2):
        box2 = box2.transpose()
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                    (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
        return inter_area / box2_area

    def __call__(self, img, target):
        h, w = img.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))
            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            # apply random color mask
            img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
            # return unobscured labels
            if len(target) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = self._bbox_ioa(box, target[:, 1:5])  # intersection over area
                target = target[ioa < 0.60]  # remove >60% obscured target
        return target
