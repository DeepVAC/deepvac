import math
import cv2
import numpy as np
import random
from .base_aug import CvAugBase

### yolov5 dataset aug
class HSVAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(HSVAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.hsv_hgain = addUserConfig('dark_gamma', self.config.dark_gamma, 0.015)
        self.config.hsv_sgain = addUserConfig('dark_gamma', self.config.dark_gamma, 0.7)
        self.config.hsv_vgain = addUserConfig('dark_gamma', self.config.dark_gamma, 0.4)

    def __call__(self, img):
        img, label = self.auditInput(img, has_label=True)
        assert isinstance(label, np.ndarray) and label.ndim == 2, "label must be numpy.ndarray, and shape should be (n, 5)"
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        # 随机增幅
        r = np.random.uniform(-1, 1, 3) * [self.config.hsv_hgain, self.config.hsv_sgain, self.config.hsv_vgain] + 1
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, label

class YoloHFlipAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(YoloHFlipAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        img, label = self.auditInput(img, has_label=True)
        assert isinstance(label, np.ndarray) and label.ndim == 2, "label must be numpy.ndarray, and shape should be (n, 5)"
        img = np.fliplr(img)
        if label.size:
            label[:, 1] = 1 - label[:, 1]
        return img, label

class YoloVFlipAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(YoloVFlipAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        img, label = self.auditInput(img, has_label=True)
        assert isinstance(label, np.ndarray) and label.ndim == 2, "label must be numpy.ndarray, and shape should be (n, 5)"
        img = np.flipud(img)
        if label.size:
            label[:, 2] = 1 - label[:, 2]
        return img, label

class YoloPerspectiveAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(YoloPerspectiveAug, self).__init__(deepvac_config)
        self.border = deepvac_config.border

    def auditConfig(self):
        self.config.yolo_perspective_scale = addUserConfig('yolo_perspective_scale', self.config.yolo_perspective_scale, 0.5)
        self.config.yolo_perspective_shear = addUserConfig('yolo_perspective_shear', self.config.yolo_perspective_shear, 0.0)
        self.config.yolo_perspective_degrees = addUserConfig('yolo_perspective_degrees', self.config.yolo_perspective_degrees, 0.0)
        self.config.yolo_perspective_translate = addUserConfig('yolo_perspective_translate', self.config.yolo_perspective_translate, 0.1)
        self.config.yolo_perspective_perspective = addUserConfig('yolo_perspective_perspective', self.config.yolo_perspective_perspective, 0.0)

    def _box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)

    def __call__(self, img):
        img, label = self.auditInput(img, has_label=True)
        assert isinstance(label, np.ndarray) and label.ndim == 2, "label must be numpy.ndarray, and shape should be (n, 5)"

        border = self.border
        h, w, c = img.shape

        width = int(w + border[1] * 2)
        height = int(h + border[0] * 2)
        # Center
        '''
            [[1, 0, -w/2],
             [0, 1, -h/2],
             [0, 0, 1.  ]]
        '''
        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2
        # Perspective
        '''
            [[1, 0, 0],
             [0, 1, 0],
             [p, p, 1]]
        '''
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.config.yolo_perspective_perspective, self.config.yolo_perspective_perspective)
        P[2, 1] = random.uniform(-self.config.yolo_perspective_perspective, self.config.yolo_perspective_perspective)
        # Rotation and Scale
        '''
            [[r, r, r],
             [r, r, r],
             [0, 0, 1]]
        '''
        R = np.eye(3)
        a = random.uniform(-self.config.yolo_perspective_degrees, self.config.yolo_perspective_degrees)
        s = random.uniform(1 - self.config.yolo_perspective_scale, 1 + self.config.yolo_perspective_scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        # Shear
        '''
            [[1, s, 0],
             [s, 1, 0],
             [0, 0, 1]]
        '''
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.config.yolo_perspective_shear, self.config.yolo_perspective_shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.config.yolo_perspective_shear, self.config.yolo_perspective_shear) * math.pi / 180)
        # Translation
        '''
            [[1, 0, t],
             [0, 1, t],
             [0, 0, 1]]
        '''
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.config.yolo_perspective_translate, 0.5 + self.config.yolo_perspective_translate) * width
        T[1, 2] = random.uniform(0.5 - self.config.yolo_perspective_translate, 0.5 + self.config.yolo_perspective_translate) * height
        # Combined rotation matrix
        M = T @ S @ R @ P @ C
        # img augment and resize to img_size
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if self.config.yolo_perspective_perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        n = len(label)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            '''
                [[x1, y1],
                 [x2, y2],
                 [x1, y2],
                 [x2, y1]]
            '''
            xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T
            if self.config.yolo_perspective_perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
            else:
                xy = xy[:, :2].reshape(n, 8)
            # x: [[x1, x2, x1, x2], ...n...]
            x = xy[:, [0, 2, 4, 6]]
            # y: [[y1, y2, y2, y1], ...n...]
            y = xy[:, [1, 3, 5, 7]]
            # xy: [[xmin, ymin, xmax, ymax], ...n...]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            # filter candidates
            i = self._box_candidates(box1=label[:, 1:5].T * s, box2=xy.T)
            label = label[i]
            label[:, 1:5] = xy[i]
        return img, label


class YoloNormalizeAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(YoloNormalizeAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, img):
        '''
            1. [cls, x1, y1, x2, y2] -> [cls, cx, cy, w, h]
            2. cx, w normalized by img width, cy, h normalized by img height
        '''
        img, label = self.auditInput(img, has_label=True)
        assert isinstance(label, np.ndarray) and label.ndim == 2, "label must be numpy.ndarray, and shape should be (n, 5)"
        if not label.size:
            return img, label
        label[:, [3, 4]] -= label[:, [1, 2]]
        label[:, [1, 2]] += label[:, [3, 4]] / 2
        label[:, [1, 3]] /= img.shape[1]
        label[:, [2, 4]] /= img.shape[0]
        return img, label
