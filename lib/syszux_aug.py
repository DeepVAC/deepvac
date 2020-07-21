import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

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

class AffineAug(AugBase):
    def __init__(self, deepvac_config):
        super(AffineAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.borderValue = (255,255,255)

    def __call__(self, img):
        rows,cols=img.shape[:2]
        shear_x = float(np.random.randint(-30,31))/100
        shear_y = float(np.random.randint(-3,4))/100
        M = np.float32([[1.0, shear_x ,0.0],[shear_y, 1.0,0.0]])
        img_affine = cv2.warpAffine(img,M,(cols,rows),borderValue=self.borderValue)
        return img_affine

class PerspectAug(AugBase):
    def __init__(self, deepvac_config):
        super(PerspectAug,self).__init__(deepvac_config)

    def auditConfig(self):
        self.borderValue = (255,255,255)

    def __call__(self, img):
        h,w=img.shape[:2]
        scale_h = np.random.randint(6,11)
        scale_w = np.random.randint(20,31)
        dirc = np.random.randint(0,4)
        point1 = np.array([[0,0],[w,0],[0,h],[w,h]],dtype = "float32")
        
        if dirc == 0:
            point2 = np.array([[w/scale_w,0],[(scale_w-1)*w/scale_w,0],[0,h],[w,h]],dtype = "float32")
        elif dirc == 1:
            point2 = np.array([[0,h/scale_h],[w,0],[0,(scale_h-1)*h/scale_h],[w,h]],dtype = "float32")
        elif dirc == 2:
            point2 = np.array([[0,0],[w,0],[w/scale_w,h],[(scale_w-1)*w/scale_w,h]],dtype = "float32")
        else:
            point2 = np.array([[0,0],[w,h/scale_h],[0,h],[w,(scale_h-1)*h/scale_h]],dtype = "float32")
        M = cv2.getPerspectiveTransform(point1,point2)
        img_perspect = cv2.warpPerspective(img,M,(w,h),borderValue=self.borderValue)
        return img_perspect
