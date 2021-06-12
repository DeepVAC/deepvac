import os
import numpy as np
import cv2
from ..utils import LOG
from .base_dataset import DatasetBase

class CocoCVDataset(DatasetBase):
    def __init__(self, deepvac_config, sample_path, target_path):
        super(CocoCVDataset, self).__init__(deepvac_config)
        try:
            from pycocotools.coco import COCO
        except:
            raise Exception("pycocotools module not found, you should try 'pip3 install pycocotools' first!")
        self.sample_path = sample_path
        self.coco = COCO(target_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cats = list(sorted(self.coco.cats.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample, label = self._getSample(index)
        sample = self.compose(sample)
        # post process
        img, label = self._buildSample(sample, label)
        return img, label

    def _getSample(self, index):
        img = self.loadImgs(index)
        category_ids, boxes, masks = self.loadAnns(index)
        # return target you want
        return img, category_ids

    def _buildSample(self, img, target):
        # BGR -> RGB && (H, W, C) -> (C, H, W)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # numpy -> tensor
        img = torch.from_numpy(img)
        target = torch.from_numpy(target)
        return img, target

    def loadImgs(self, index):
        file_name = self.coco.loadImgs(self.ids[index])[0]['file_name']
        img = cv2.imread(os.path.join(self.sample_path, file_name), 1)
        assert img is not None, "Image {} not found!".format(file_name)
        return img

    def loadAnns(self, index):
        ann_ids = self.coco.getAnnIds(imgIds=self.ids[index])
        anns = self.coco.loadAnns(ann_ids)
        category_ids = np.array([self.cats.index(i["category_id"]) for i in anns], dtype=np.float)
        boxes = np.array([i["bbox"] for i in anns], dtype=np.float)
        masks = np.array([self.coco.annToMask(i) for i in anns], dtype=np.float)
        return category_ids, boxes, masks


class CocoCVSegDataset(DatasetBase):
    def __init__(self, deepvac_config, sample_path_prefix, target_path, cat2idx):
        super(CocoCVSegDataset, self).__init__(deepvac_config)
        try:
            from pycocotools.coco import COCO
        except:
            raise Exception("pycocotools module not found, you should try 'pip3 install pycocotools' first!")
        self.sample_path_prefix = sample_path_prefix
        self.coco = COCO(target_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cats = list(sorted(self.coco.cats.keys()))
        self.cat2idx = cat2idx
        LOG.logI("Notice: 0 will be treated as background in {}!!!".format(self.name()))

    def auditConfig(self):
        self.auto_detect_subdir_with_basenum = self.addUserConfig('auto_detect_subdir_with_basenum', self.config.auto_detect_subdir_with_basenum, 0)
        LOG.logI("You set auto_detect_subdir_with_basenum to {}".format(self.auto_detect_subdir_with_basenum))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        sample, mask, cls_masks, file_path = self._getSample(id)
        sample, mask, cls_masks, file_path = self.compose((sample, mask, cls_masks, os.path.join(self.sample_path_prefix, file_path)))
        return sample, mask, file_path

    def updatePath(self, id, file_path):
        full_file_path = self.coco.loadImgs(id)[0]["path"]
        path_list = full_file_path.split('/')
        path_list_num = len(path_list)

        if path_list_num == self.auto_detect_subdir_with_basenum:
            withsub_file_path = file_path
        elif path_list_num == self.auto_detect_subdir_with_basenum + 1:
            withsub_file_path = path_list[-2] + '/' + file_path
            file_path = path_list[-2] + '_' + file_path
        else:
            LOG.logE("path list has {} fields, which should be {} or {}".format(path_list_num, self.auto_detect_subdir_with_basenum, self.auto_detect_subdir_with_basenum+1), exit=True)

        return withsub_file_path, file_path

    def _getSample(self, id: int):
        # img
        file_path = self.coco.loadImgs(id)[0]["file_name"]
        withsub_file_path = file_path
        if self.auto_detect_subdir_with_basenum > 0:
            withsub_file_path, file_path = self.updatePath(id, file_path)

        img = cv2.imread(os.path.join(self.sample_path_prefix, withsub_file_path), 1)
        assert img is not None, "Image {} not found in {} !".format(withsub_file_path, self.sample_path_prefix)
        # anno
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        cls_masks = {}
        h, w = img.shape[:2]
        mask = np.zeros((h, w))
        bg_mask = None
        for idx, ann in enumerate(anns):
            # current category as 1, BG and others as 0.
            cls_mask = self.coco.annToMask(ann)
            cls_idx = self.cat2idx[ann["category_id"]]
            cls_masks[cls_idx] = cls_mask
            mask[cls_mask==1] = cls_idx

        # return target you want
        return img, mask, cls_masks, file_path
