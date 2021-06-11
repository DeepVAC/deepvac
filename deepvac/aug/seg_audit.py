import os
import cv2
import numpy as np
from ..utils import LOG, addUserConfig
from .base_aug import CvAugBase

class CvAugSegAuditBase4(CvAugBase):
    def __init__(self, deepvac_config):
        super(CvAugSegAuditBase4, self).__init__(deepvac_config)
        self.input_len = self.addUserConfig('input_len', self.config.input_len, 4)
        self.cls_num = self.addUserConfig('cls_num', self.config.cls_num, 4)
        self.pallete = [[255, 255, 255],
            [255, 0,  0],
            [0, 255,  0],
            [0,  0,  255],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70,  130, 180],
            [220, 20,  60],
            [255, 0,   0],
            [0,   0,   142],
            [0,   0,   70],
            [0,   60,  100],
            [0,   80,  100],
            [0,   0,   230],
            [119, 11,  32]]

    def putMask(self, img, mask):
        h,w = img.shape[:2]
        classMap_numpy_color = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(self.cls_num):
            [r, g, b] = self.pallete[idx]
            classMap_numpy_color[mask == idx] = [b, g, r]
        overlayed = cv2.addWeighted(img, 0.5, classMap_numpy_color, 0.5, 0)
        return overlayed

    def getBaseName(self, fn):
        return os.path.basename(fn)

    def write(self, img, label, fn):
        overlayed = self.putMask(img, label)
        basename = self.getBaseName(fn)
        cv2.imwrite(os.path.join(self.save_dir, os.path.splitext(basename)[0]+'.png'), overlayed)

class ImageWithMaskIntersectAudit(CvAugSegAuditBase4):
    def auditConfig(self):
        self.intersect_ratio = self.addUserConfig('intersect_ratio', self.config.intersect_ratio, 0.10)
        self.remask = self.addUserConfig('remask', self.config.remask, True)
        self.save_dir = self.addUserConfig('intersect_dir', self.config.intersect_dir, "intersect_dir")
        self.is_consider_bg = self.addUserConfig('is_consider_bg', self.config.is_consider_bg, False)
        os.makedirs(self.save_dir, exist_ok=True)
    
    def remaskIfNeeded(self, i, j):
        mask_i = self.cls_masks[self.catid[i]]
        mask_j = self.cls_masks[self.catid[j]]
        mask_i_num = np.sum(mask_i==1)
        mask_j_num = np.sum(mask_j==1)
        min_area = min(mask_i_num, mask_j_num)
        
        intersect_area = mask_i*mask_j==1
        intersect_ratio = np.sum(intersect_area) / min_area

        if intersect_ratio < self.intersect_ratio:
            return

        is_write = self.catid[i] * self.catid[j] != 0
        if is_write or self.is_consider_bg:
            LOG.logI("Image {} has risk on rule {} with value {} ({} / {})".format(self.fn, self.name(), intersect_ratio, self.catid[i], self.catid[j]))

        if self.remask:
            min_idx, max_idx = (i,j) if mask_i_num < mask_j_num else (j,i)
            LOG.logI("You have enabled remask, do remask...")
            self.label[intersect_area] = self.catid[min_idx]
            self.cls_masks[self.catid[max_idx]][intersect_area] = self.catid[min_idx]

        #if bg
        if is_write or self.is_consider_bg:
            self.write(self.img, self.label, self.fn)

    def forward(self, imgs):
        self.img, self.label, self.cls_masks, self.fn = imgs
        len_mask = len(self.cls_masks)
        if len_mask < 2:
            return imgs
        self.catid = list(self.cls_masks.keys())
        for i in range(len_mask - 1):
            for j in range(i+1, len_mask):
                self.remaskIfNeeded(i, j)

        return self.img, self.label, self.cls_masks, self.fn

class ImageWithMaskSideRatioAudit(CvAugSegAuditBase4):
    def auditConfig(self):
        self.save_dir = self.addUserConfig('side_ratio_dir', self.config.side_ratio_dir, "side_ratio_dir")
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, imgs):
        img, label, _, fn = imgs
        if img.shape[0] == img.shape[1]:
            LOG.logI("Image {} has risk on rule {} with size {}".format(fn, self.name(), img.shape))
            self.write(img, label, fn)
        return imgs

class ImageWithMaskTargetSizeAudit(CvAugSegAuditBase4):
    def auditConfig(self):
        self.min_ratio = self.addUserConfig('min_ratio', self.config.min_ratio, 900)
        self.save_dir = self.addUserConfig('target_size_dir', self.config.target_size_dir, "target_size_dir")
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, imgs):
        img, label, cls_masks, fn = imgs
        min_pixel_num = int(label.shape[0] * label.shape[1] / self.min_ratio)
        for cls_idx in cls_masks.keys():
            cls_pixel_num = np.sum(label==cls_idx)
            if cls_pixel_num == 0:
                continue
            if cls_pixel_num >= min_pixel_num:
                continue
            LOG.logI("Image {} has risk on rule {} with {}_num = {} (min_pixel_num = {})".format(fn, self.name(), cls_idx, cls_pixel_num, min_pixel_num))
            self.write(img, label, fn)
        return imgs

class ImageWithMaskVisionAudit(CvAugSegAuditBase4):
    def auditConfig(self):
        self.save_dir = self.addUserConfig('vision_dir', self.config.vision_dir, "vision_dir")
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, imgs):
        img, label, _, fn = imgs
        self.write(img, label, fn)

        label_path = os.path.dirname(fn)+'_label'
        basename = self.getBaseName(fn)
        cv2.imwrite(os.path.join(label_path, os.path.splitext(basename)[0]+'.png'), label)
        return imgs


