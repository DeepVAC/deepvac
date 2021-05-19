import cv2
import numpy as np
import random
import torch
from .base_aug import CvAugBase

class ImageWithMasksRandomHorizontalFlipAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(ImageWithMasksRandomHorizontalFlipAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        imgs = [img]
        imgs.extend(label)
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
        return [imgs[0],imgs[1:]]

class ImageWithMasksRandomRotateAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(ImageWithMasksRandomRotateAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.max_angle = self.addUserConfig('max_angle', self.config.max_angle, 10)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        imgs = [img]
        imgs.extend(label)
        angle = random.random() * 2 * self.config.max_angle - self.config.max_angle
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = img_rotation
        return [imgs[0],imgs[1:]]

class ImageWithMasksRandomCropAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(ImageWithMasksRandomCropAug, self).__init__(deepvac_config)
        self.auditUserConfig("img_size")

    def auditConfig(self):
        self.config.p = self.addUserConfig('p', self.config.p, 3.0 / 8.0)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, input_len=2)
        h, w, _ = img.shape

        imgs = [img]
        imgs.extend(label)
        if max(h, w) <= self.config.img_size:
            return [imgs[0],imgs[1:]]

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
        return [imgs[0],imgs[1:]]

class ImageWithMasksScaleAug(CvAugBase):
    def auditConfig(self):
        self.config.w = self.addUserConfig('w', self.config.w, 384, True)
        self.config.h = self.addUserConfig('h', self.config.h, 384, True)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs)
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]

class ImageWithMasksRandomCropResizeAug(CvAugBase):
    def auditConfig(self):
        self.config.size = self.addUserConfig('size', self.config.size, 384, True)
        self.config.max_crop_ratio = self.addUserConfig('max_crop_ratio', self.config.max_crop_ratio, 0.1)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs)
        h, w = img.shape[:2]
        x1 = random.randint(0, int(w*self.config.max_crop_ratio)) # 25% to 10%
        y1 = random.randint(0, int(h*self.config.max_crop_ratio))

        img_crop = img[y1:h-y1, x1:w-x1]
        label_crop = label[y1:h-y1, x1:w-x1]

        img_crop = cv2.resize(img_crop, self.config.size)
        label_crop = cv2.resize(label_crop, self.config.size, interpolation=cv2.INTER_NEAREST)
        return [img_crop, label_crop]

class ImageWithMasksHFlipAug(CvAugBase):
    def __call__(self, imgs):
        img, label = self.auditInput(imgs)
        image = cv2.flip(image, 1) # horizontal flip
        label = cv2.flip(label, 1) # horizontal flip
        return [image, label]

class ImageWithMasksVFlipAug(CvAugBase):
    def __call__(self, imgs):
        img, label = self.auditInput(imgs)
        image = cv2.flip(image, 0) # veritcal flip
        label = cv2.flip(label, 0)  # veritcal flip
        return [image, label]

class ImageWithMasksNormalizeAug(CvAugBase):
    def auditConfig(self):
        self.config.mean = self.addUserConfig('mean', self.config.mean, 0, True)
        self.config.std = self.addUserConfig('std', self.config.std, 0, True)

    def __call__(self, imgs):
        image, label = self.auditInput(imgs)
        image = image.astype(np.float32)
        for i in range(3):
            image[:,:,i] -= self.config.mean[i]
        for i in range(3):
            image[:,:, i] /= self.config.std[i]
        return [image, label]

class ImageWithMasksToTensorAug(CvAugBase):
    def auditConfig(self):
        self.config.scale = self.addUserConfig('scale', self.config.scale, 1)

    def __call__(self, imgs):
        image, label = self.auditInput(imgs)

        if self.config.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.config.scale), int(h/self.config.scale)), interpolation=cv2.INTER_NEAREST)

        default_float_dtype = torch.get_default_dtype()
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(image_tensor, torch.ByteTensor):
            image_tensor = image_tensor.to(dtype=default_float_dtype).div(255)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int)) #torch.from_numpy(label)

        return [image_tensor, label_tensor]
