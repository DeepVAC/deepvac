import cv2
import numpy as np
import random
from .base_aug import CvAugBase

# 随机旋转（针对于人脸关键点任务）
class RandomRotateFacialKpListAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(RandomRotateFacialKpListAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    # img: include 2 element
    # img[0] -> img
    # img[1] -> keypoint info (scale to 0-1)
    def __call__(self, img):
        img,landmarks = self.auditInput(img, has_label=True)
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
class RandomFilpFacialKpListAug(CvAugBase):
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

    # img: include 2 element
    # img[0] -> img
    # img[1] -> keypoint info(scale to 0-1)
    def __call__(self, img):
        img,landmarks = self.auditInput(img, has_label=True)
        h, w, _ = img.shape
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

class CropFacialWithBoxesAndLmksAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(CropFacialWithBoxesAndLmksAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.facial_img_dim = addUserConfig('facial_img_dim', self.config.facial_img_dim, 640, True)

    def _matrix_iof(self, a, b):
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / np.maximum(area_a[:, np.newaxis], 1)

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        boxes, landms, labels = label
        height, width, _ = image.shape

        for _ in range(250):

            PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
            scale = random.choice(PRE_SCALES)
            short_side = min(width, height)
            w = int(scale * short_side)
            h = w

            if width == w:
                l = 0
            else:
                l = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            value = self._matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask_a].copy()
            labels_t = labels[mask_a].copy()
            landms_t = landms[mask_a].copy()
            landms_t = landms_t.reshape([-1, 5, 2])

            if boxes_t.shape[0] == 0:
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            # landm
            landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
            landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
            landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
            landms_t = landms_t.reshape([-1, 10])


            # make sure that the cropped image contains at least one face > 16 pixel at training image scale
            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * self.config.facial_img_dim
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * self.config.facial_img_dim
            mask_b = np.minimum(b_w_t, b_h_t) > 0.0
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]
            landms_t = landms_t[mask_b]

            if boxes_t.shape[0] == 0:
                continue

            return image_t, [boxes_t, landms_t, labels_t]
        return image, [boxes, landms, labels]


class DistortFacialAugBase(CvAugBase):
    def __init__(self, deepvac_config):
        super(DistortFacialAugBase, self).__init__(deepvac_config)

    def _convert(self, image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    def auditConfig(self):
        pass

class BrightDistortFacialAug(DistortFacialAugBase):
    def __init__(self, deepvac_config):
        super(BrightDistortFacialAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        self._convert(image, beta=random.uniform(-32, 32))

        return image, label

class ContrastDistortFacialAug(DistortFacialAugBase):
    def __init__(self, deepvac_config):
        super(ContrastDistortFacialAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        self._convert(image, alpha=random.uniform(0.5, 1.5))

        return image, label

class SaturationDistortFacialAug(DistortFacialAugBase):
    def __init__(self, deepvac_config):
        super(SaturationDistortFacialAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self._convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image, label

class HueDistortFacialAug(DistortFacialAugBase):
    def __init__(self, deepvac_config):
        super(HueDistortFacialAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image, label

class MirrorFacialAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(MirrorFacialAug, self).__init__(deepvac_config)

    def auditConfig(self):
        pass

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        boxes, landms, labels = label
        _, width, _ = image.shape
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

        return image, [boxes, landms, labels]

class Pad2SquareFacialAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(Pad2SquareFacialAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.facial_rgb_means = addUserConfig('facial_rgb_means', self.config.facial_rgb_means, (104, 117, 123))

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        height, width, _ = image.shape
        if height == width:
            return image, label
        long_side = max(width, height)
        image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
        image_t[:, :] = self.config.facial_rgb_means
        image_t[0:0 + height, 0:0 + width] = image
        return image, label

class ResizeSubtractMeanFacialAug(CvAugBase):
    def __init__(self, deepvac_config):
        super(ResizeSubtractMeanFacialAug, self).__init__(deepvac_config)

    def auditConfig(self):
        self.config.facial_img_dim = addUserConfig('facial_img_dim', self.config.facial_img_dim, 640, True)
        self.config.facial_rgb_means = addUserConfig('facial_rgb_means', self.config.facial_rgb_means, (104, 117, 123))

    def __call__(self, image):
        image, label = self.auditInput(image, has_label=True)
        assert isinstance(label, list) and len(label) == 3, "label must be list, and length should be 3"
        assert isinstance(label[0], np.ndarray) and label[0].ndim == 2, "label[0](boxes) must be numpy.ndarray, and shape should be (n, 4)"
        assert isinstance(label[1], np.ndarray) and label[1].ndim == 2, "label[1](landms) must be numpy.ndarray, and shape should be (n, 10)"
        assert isinstance(label[2], np.ndarray) and label[2].ndim == 1, "label[2](labels) must be numpy.ndarray, and shape should be (n, )"

        boxes, landms, labels = label
        height, width, _ = image.shape

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, (self.config.facial_img_dim, self.config.facial_img_dim), interpolation=interp_method)
        image = image.astype(np.float32)
        image -= self.config.facial_rgb_means

        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        landms[:, 0::2] /= width
        landms[:, 1::2] /= height

        return image.transpose(2, 0, 1), [boxes, landms, labels]