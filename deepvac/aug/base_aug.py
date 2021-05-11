import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
from scipy import ndimage
from ..utils import LOG
from .warp_mls_helper import WarpMLS

class AugBase(object):
    def __init__(self, deepvac_aug_config):
        self.config = deepvac_aug_config
        self.auditConfig()

    def auditConfig(self):
        raise Exception("Not implemented!")

    def addUserConfig(self, config_name, user_give=None, developer_give=None, is_user_mandatory=False):
        if user_give is None and is_user_mandatory:
            LOG.logE("You must set mandatory config.aug.{} in config.py. Developer advised value:{}".format(config_name, developer_give),exit=True)
        if user_give:
            return user_give
        if developer_give:
            return developer_give
        LOG.logE("value missing for configuration: {}".format(config_name), exit=True)

    def __call__(self,img):
        raise Exception("Not implemented!")

    @staticmethod
    def auditInput(img, has_label=False):
        if not has_label:
            assert isinstance(img, np.ndarray) and img.ndim == 3, "image must be cv2 image"
            return

        if not isinstance(img, (list, tuple)):
            assert False, "parameter img must be [image, labels]"
            return

        assert len(img) == 2, "parameter img must be [image, labels]"
        img, label = img
        assert isinstance(img, np.ndarray) and img.ndim == 3, "image must be cv2 image"
        return img, label

    @staticmethod
    def pillow2cv(pillow_img, is_rgb2bgr=True):
        cv_image = np.array(pillow_img)
        if is_rgb2bgr:
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
        self.config.speckle_severity = self.addUserConfig('speckle_severity', self.config.speckle_severity, np.random.uniform(0, 0.6*255))

    def __call__(self, img):
        input_type = img.dtype
        self.auditInput(img)
        blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * self.config.speckle_severity, 1)
        img_speck = (img + blur)
        img_speck[img_speck > 255] = 255
        img_speck[img_speck <= 0] = 0
        img_speck = img_speck.astype(input_type)
        return img_speck

# 仿射变换
class AffineAug(AugBase):
    def __init__(self, deepvac_config):
        super(AffineAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # 空白填充色
        self.config.affine_borderValue = self.addUserConfig('affine_borderValue', self.config.affine_borderValue, (255,255,255))
        # x方向和y方向的伸缩率
        self.config.affine_shear_x = self.addUserConfig('affine_shear_x', self.config.affine_shear_x, 30)
        self.config.affine_shear_y = self.addUserConfig('affine_shear_y', self.config.affine_shear_y, 1)

    def __call__(self, img):
        self.auditInput(img)
        rows,cols=img.shape[:2]
        shear_x = float(np.random.randint(-self.config.affine_shear_x, self.config.affine_shear_x + 1))/100
        shear_y = float(np.random.randint(-self.config.affine_shear_y, self.config.affine_shear_y + 1))/100
        M = np.float32([[1.0, shear_x ,0.0],[shear_y, 1.0,0.0]])
        img_affine = cv2.warpAffine(img,M,(cols,rows),borderValue=self.config.affine_borderValue)
        return img_affine

# 透视变换
class PerspectAug(AugBase):
    def __init__(self, deepvac_config):
        super(PerspectAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # b空白填充色
        self.config.perspect_borderValue = self.addUserConfig('perspect_borderValue', self.config.perspect_borderValue, (255,255,255))
        # 高h方向伸缩范围
        self.config.perspect_sh_lower = self.addUserConfig('perspect_sh_lower', self.config.perspect_sh_lower, 6)
        self.config.perspect_sh_upper = self.addUserConfig('perspect_sh_upper', self.config.perspect_sh_upper, 11)
        # 宽w方向伸缩范围
        self.config.perspect_sw_lower = self.addUserConfig('perspect_sw_lower', self.config.perspect_sw_lower, 20)
        self.config.perspect_sw_upper = self.addUserConfig('perspect_sw_upper', self.config.perspect_sw_upper, 31)

    def __call__(self, img):
        self.auditInput(img)
        h,w=img.shape[:2]
        scale_h = np.random.randint(self.config.perspect_sh_lower ,self.config.perspect_sh_upper)
        scale_w = np.random.randint(self.config.perspect_sw_lower ,self.config.perspect_sw_upper)
        point1 = np.array([[0,0],[w,0],[0,h],[w,h]],dtype = "float32")
        point2_list = [
            np.array([[w/scale_w,0],[(scale_w-1)*w/scale_w,0],[0,h],[w,h]],dtype = "float32"),
            np.array([[0,h/scale_h],[w,0],[0,(scale_h-1)*h/scale_h],[w,h]],dtype = "float32"),
            np.array([[0,0],[w,0],[w/scale_w,h],[(scale_w-1)*w/scale_w,h]],dtype = "float32"),
            np.array([[0,0],[w,h/scale_h],[0,h],[w,(scale_h-1)*h/scale_h]],dtype = "float32")]

        pt_idx = np.random.randint(0,4)
        M = cv2.getPerspectiveTransform(point1,point2_list[pt_idx])
        img_perspect = cv2.warpPerspective(img,M,(w,h),borderValue=self.config.perspect_borderValue)
        return img_perspect

# 高斯模糊
class GaussianAug(AugBase):
    def __init__(self, deepvac_config):
        super(GaussianAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.gaussian_ks = self.addUserConfig('gaussian_ks', self.config.gaussian_ks, [9,11,13,15,17])

    def __call__(self, img):
        self.auditInput(img)
        ks = self.config.gaussian_ks[np.random.randint(0,len(self.config.gaussian_ks))]
        img_gaussian = cv2.GaussianBlur(img,(ks, ks), 0)
        return img_gaussian

# 添加横线增强
class HorlineAug(AugBase):
    def __init__(self, deepvac_config):
        super(HorlineAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # 线条间隔
        self.config.horline_space = self.addUserConfig('horline_space', self.config.horline_space, 4)
        # 线条颜色
        self.config.horline_color = self.addUserConfig('horline_color', self.config.horline_color, 0)
        # 线宽
        self.config.horline_thickness = self.addUserConfig('horline_thickness', self.config.horline_thickness, 1)

    def __call__(self, img):
        self.auditInput(img)
        img_h, img_w = img.shape[:2]
        for i in range(0,img_h,self.config.horline_space):
            cv2.line(img, (0, i), (img_w, i), self.config.horline_color, self.config.horline_thickness)
        return img

# 添加竖线增强
class VerlineAug(AugBase):
    def __init__(self, deepvac_config):
        super(VerlineAug,self).__init__(deepvac_config)

    def auditConfig(self):
        # 线条间隔
        self.config.verline_space = self.addUserConfig('verline_space', self.config.verline_space, 4)
        # 线条颜色
        self.config.verline_color = self.addUserConfig('verline_color', self.config.verline_color, 0)
        # 线宽
        self.config.verline_thickness = self.addUserConfig('verline_thickness', self.config.verline_thickness, 1)

    def __call__(self, img):
        self.auditInput(img)
        img_h, img_w = img.shape[:2]
        for i in range(0,img_w,self.config.verline_space):
            cv2.line(img, (i, 0), (i, img_h), self.config.verline_color, self.config.verline_thickness)
        return img

# 左右运动模糊
class LRmotionAug(AugBase):
    def __init__(self, deepvac_config):
        super(LRmotionAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.lrmotion_ks = self.addUserConfig('lrmotion_ks', self.config.lrmotion_ks, [3,5,7,9])

    def __call__(self,img):
        self.auditInput(img)
        ks = self.config.lrmotion_ks[np.random.randint(0,len(self.config.lrmotion_ks))]
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
        self.config.udmotion_ks = self.addUserConfig('udmotion_ks', self.config.udmotion_ks, [3,5,7,9])

    def __call__(self, img):
        self.auditInput(img)
        ks = self.config.udmotion_ks[np.random.randint(0,len(self.config.udmotion_ks))]
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
        self.config.noisy_mean = self.addUserConfig('noisy_mean', self.config.noisy_mean, 0)
        self.config.noisy_sigma = self.addUserConfig('noisy_sigma', self.config.noisy_sigma, 1)

    def __call__(self, img):
        self.auditInput(img)
        row, col = img.shape[:2]
        gauss = np.random.normal(self.config.noisy_mean, self.config.noisy_sigma, (row, col,3))
        gauss = gauss.reshape(row, col,3)
        noisy = img + gauss
        img_noisy= noisy.astype(np.uint8)
        return img_noisy

# 扭曲变换
class DistortAug(AugBase):
    def __init__(self, deepvac_config):
        super(DistortAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.distort_segment = self.addUserConfig('distort_segment', self.config.distort_segment, 4)

    def __call__(self, img):
        self.auditInput(img)
        img_h, img_w = img.shape[:2]
        cut = img_w // self.config.distort_segment
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
        for cut_idx in np.arange(1, self.config.distort_segment, 1):
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
        self.config.stretch_segment = self.addUserConfig('stretch_segment', self.config.stretch_segment, 4)

    def __call__(self, img):
        self.auditInput(img)
        img_h, img_w = img.shape[:2]

        cut = img_w // self.config.stretch_segment
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

        for cut_idx in np.arange(1, self.config.stretch_segment, 1):
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
        self.auditInput(img)
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
        self.config.motion_degree = self.addUserConfig('motion_degree', self.config.motion_degree, 18)
        self.config.motion_angle = self.addUserConfig('motion_angle', self.config.motion_angle, 45)

    def __call__(self, img):
        self.auditInput(img)
        m = cv2.getRotationMatrix2D((self.config.motion_degree / 2, self.config.motion_degree / 2), self.config.motion_angle, 1)
        motion_blur_kernel = np.diag(np.ones(self.config.motion_degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (self.config.motion_degree, self.config.motion_degree))
        motion_blur_kernel = motion_blur_kernel / self.config.motion_degree
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)

        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        return blurred

# 降低图片亮度
class DarkAug(AugBase):
    def __init__(self, deepvac_config):
        super(DarkAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.dark_gamma = self.addUserConfig('dark_gamma', self.config.dark_gamma, 3)

    def __call__(self, img):
        self.auditInput(img)
        is_gray = img.ndim == 2 or img.shape[1] == 1
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        illum = np.power(illum, self.config.dark_gamma)
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
        self.config.halfdark_gamma = self.addUserConfig('halfdark_gamma', self.config.halfdark_gamma, 1.5)

    def __call__(self, img):
        self.auditInput(img)
        h, w, _ = img.shape
        is_gray = img.ndim == 2 or img.shape[1] == 1
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        illum = hsv[..., 2] / 255.
        illum[:, w//2:] = np.power(illum[:, w//2:], self.config.halfdark_gamma)
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
        self.auditInput(img)
        half_dark = HalfDarkAug(self.deepvac_config)
        half_darked = half_dark(img)
        motion = MotionAug(self.deepvac_config)
        motioned = motion(half_darked)
        return motioned

# 随机crop框降低亮度
class RandomCropDarkAug(AugBase):
    def __init__(self, deepvac_config):
        super(RandomCropDarkAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.random_crop_dark_gamma = self.addUserConfig('random_crop_dark_gamma', self.config.random_crop_dark_gamma, 1.2)

    def __call__(self, img):
        self.auditInput(img)
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
        illum = np.power(illum, self.config.random_crop_dark_gamma)
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

        LOG.logI('rect in RandomCropDarkAug:{}'.format(rect))
        return img

# 随机颜色扰动
class ColorJitterAug(AugBase):
    def __init__(self, deepvac_config):
        super(ColorJitterAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
        img = self.cv2pillow(img)
        img = ImageEnhance.Color(img).enhance(np.random.uniform(0.8, 1.3))
        return self.pillow2cv(img)

class BrightnessJitterAug(AugBase):
    def __init__(self, deepvac_config):
        super(BrightnessJitterAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
        img = self.cv2pillow(img)
        img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.6, 1.5))
        return self.pillow2cv(img)

class ContrastJitterAug(AugBase):
    def __init__(self, deepvac_config):
        super(ContrastJitterAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
        img = self.cv2pillow(img)
        img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.5, 1.8))
        return self.pillow2cv(img)

class RandomColorJitterAug(AugBase):
    def __init__(self, deepvac_config):
        super(RandomColorJitterAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
        img = self.cv2pillow(img)
        if random.randint(0, 1):
            img = ImageEnhance.Color(img).enhance(np.random.uniform(0.8, 1.3))
        if random.randint(0, 1):
            img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.6, 1.5))
        if random.randint(0, 1):
            img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.5, 1.8))
        return self.pillow2cv(img)

class MosaicAug(AugBase):
    def __init__(self, deepvac_config):
        super(MosaicAug, self).__init__(deepvac_config)
        self.neighbor = 2

    def auditConfig(self):
        pass

    def __call__(self, img):
        self.auditInput(img)
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
