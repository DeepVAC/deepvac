import cv2
import numpy as np
import random
import torch
from .base_aug import CvAugBase

class ImageWithMasksRandomHorizontalFlipAug(CvAugBase):
    def auditConfig(self):
        self.config.input_len = self.addUserConfig('input_len', self.config.input_len, 2)

    def __call__(self, imgs):
        imgs = self.auditInput(imgs, input_len=self.config.input_len)
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
        return imgs

class ImageWithMasksRandomRotateAug(CvAugBase):
    def auditConfig(self):
        self.config.max_angle = self.addUserConfig('max_angle', self.config.max_angle, 10)
        self.config.input_len = self.addUserConfig('input_len', self.config.input_len, 2)
        self.config.label_bg_color = self.addUserConfig('label_bg_color', self.config.label_bg_color, (0, 0, 0))

    def __call__(self, imgs):
        """
        param: imgs = [img, label1, label2, ...]
        """
        imgs = self.auditInput(imgs, input_len=self.config.input_len)
        # angle
        angle = random.random() * 2 * self.config.max_angle - self.config.max_angle
        # fill color
        if isinstance(self.config.fill_color, (tuple, list)):
            fill_color = self.config.fill_color
            try:
                fill_color = [min(max(int(fill_color[i]), 0), 255) for i in range(3)]
            except:
                raise Exception("fill_color = (int, int, int) while fill_color type = list or tuple")
        elif isinstance(self.config.fill_color, bool):
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if self.config.fill_color else (0, 0, 0)

        elif self.config.fill_color is None:
            fill_color = (0, 0, 0)
        else:
            raise TypeError("fill_color type must be (None, bool, tuple, list)")
        # ops
        w, h = imgs[0].shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        imgs[0] = cv2.warpAffine(imgs[0], rotation_matrix, (h, w), borderValue=fill_color)
        for i in range(1, len(imgs)):
            img = imgs[i]
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), borderValue=self.config.label_bg_color)
            imgs[i] = img_rotation
        return imgs

class ImageWithMasksRandom4TextCropAug(CvAugBase):
    def auditConfig(self):
        self.config.p = self.addUserConfig('p', self.config.p, 3.0 / 8.0)
        self.config.img_size = self.addUserConfig('img_size', self.config.img_size, 224)
        self.config.input_len = self.addUserConfig('input_len', self.config.input_len, 2)

    def __call__(self, imgs):
        imgs = self.auditInput(imgs, input_len=self.config.input_len)
        h, w, _ = imgs[0].shape

        if max(h, w) <= self.config.img_size:
            return imgs

        img_size = (self.config.img_size,) * 2
        th = min(h, img_size[0])
        tw = min(w, img_size[0])

        if random.random() > self.config.p and np.max(imgs[1]) > 0:
            tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
            tl[tl < 0] = 0
            br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
            br[br < 0] = 0
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = random.randint(tl[0], br[0])
            j = random.randint(tl[1], br[1])
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # return i, j, th, tw
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        return imgs

class ImageWithMasksScaleAug(CvAugBase):
    def auditConfig(self):
        self.config.w = self.addUserConfig('w', self.config.w, 384, True)
        self.config.h = self.addUserConfig('h', self.config.h, 384, True)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        img = cv2.resize(img, (self.config.w, self.config.h))
        label = cv2.resize(label, (self.config.w, self.config.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]

class ImageWithMasksSafeCropAug(CvAugBase):
    def __call__(self, imgs):
       img, label = self.auditInput(imgs, input_len=2)
       h, w = img.shape[:2]

       xmin, ymin, xmax, ymax = self.getXY(label)
       x1 = random.randint(0, xmin)
       y1 = random.randint(0, ymin)
       x2 = random.randint(xmax, w)
       y2 = random.randint(ymax, h)

       img_crop = img[y1:y2, x1:x2, :]
       label_crop = label[y1:y2, x1:x2]
       return img_crop, label_crop

    def getXY(self, label):
        coord = label.nonzero()
        ymin, xmin = coord[0].min(), coord[1].min()
        ymax, xmax = coord[0].max(), coord[1].max()
        return xmin, ymin, xmax, ymax

class ImageWithMasksCenterCropAug(CvAugBase):
    def auditConfig(self):
        self.config.max_crop_ratio = self.addUserConfig('max_crop_ratio', self.config.max_crop_ratio, 0.1)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        h, w = img.shape[:2]
        x1 = random.randint(0, int(w*self.config.max_crop_ratio)) # 25% to 10%
        y1 = random.randint(0, int(h*self.config.max_crop_ratio))

        img_crop = img[y1:h-y1, x1:w-x1]
        label_crop = label[y1:h-y1, x1:w-x1]

        return [img_crop, label_crop]

class ImageWithMasksHFlipAug(CvAugBase):
    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        img = cv2.flip(img, 1) # horizontal flip
        label = cv2.flip(label, 1) # horizontal flip
        return [img, label]

class ImageWithMasksVFlipAug(CvAugBase):
    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        img = cv2.flip(img, 0) # veritcal flip
        label = cv2.flip(label, 0)  # veritcal flip
        return [img, label]

class ImageWithMasksNormalizeAug(CvAugBase):
    def auditConfig(self):
        self.config.mean = self.addUserConfig('mean', self.config.mean,  [137.98341,142.35637,150.78705], True)
        self.config.std = self.addUserConfig('std', self.config.std, [62.98702,63.34315,62.743645], True)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        img = img.astype(np.float32)
        for i in range(3):
            img[:,:,i] -= self.config.mean[i]
        for i in range(3):
            img[:,:, i] /= self.config.std[i]
        return [img, label]

class ImageWithMasksToTensorAug(CvAugBase):
    def auditConfig(self):
        self.config.scale = self.addUserConfig('scale', self.config.scale, 1)
        self.config.force_div255 = self.addUserConfig('force_div255', self.config.force_div255, True)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)

        if self.config.scale != 1:
            h, w = label.shape[:2]
            img = cv2.resize(img, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.config.scale), int(h/self.config.scale)), interpolation=cv2.INTER_NEAREST)

        default_float_dtype = torch.get_default_dtype()
        image_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(image_tensor, torch.ByteTensor) or self.config.force_div255:
            image_tensor = image_tensor.to(dtype=default_float_dtype).div(255)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int)) #torch.from_numpy(label)

        return [image_tensor, label_tensor]
