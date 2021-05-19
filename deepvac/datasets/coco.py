import os
import numpy as np
import cv2
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
        # augment for numpy
        if self.config.composer is not None:
            sample = self.config.composer(sample)
        # pytorch offical transform for PIL Image or numpy
        if self.config.transform is not None:
            # rewrite method: '_buildSample' first
            sample = self.config.transform(sample)
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
