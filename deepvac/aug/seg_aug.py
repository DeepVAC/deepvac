import cv2
import numpy as np
import random
from .base_aug import CvAugBase

class ImageWithMasksRandomHorizontalFlipAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(ImageWithMasksRandomHorizontalFlipAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, has_label=True)
        imgs = [img]
        imgs.extend(label)
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)
        return [imgs[0],imgs[1:]]

class ImageWithMasksRandomRotateAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(ImageWithMasksRandomRotateAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.image_with_masks_random_rotate_max_angle = addUserConfig('image_with_masks_random_rotate_max_angle', self.config.image_with_masks_random_rotate_max_angle, 10)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, has_label=True)
        imgs = [img]
        imgs.extend(label)
        angle = random.random() * 2 * self.config.image_with_masks_random_rotate_max_angle - self.config.image_with_masks_random_rotate_max_angle
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
        self.config.image_with_masks_random_crop_p = addUserConfig('image_with_masks_random_crop_p', self.config.image_with_masks_random_crop_p, 3.0 / 8.0)

    def __call__(self, imgs):
        img, label = self.auditInput(imgs, has_label=True)
        h, w, _ = img.shape

        imgs = [img]
        imgs.extend(label)
        if max(h, w) <= self.config.img_size:
            return [imgs[0],imgs[1:]]

        img_size = (self.config.img_size,) * 2
        th = min(h, img_size[0])
        tw = min(w, img_size[0])

        if random.random() > self.config.image_with_masks_random_crop_p and np.max(imgs[1]) > 0:
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

