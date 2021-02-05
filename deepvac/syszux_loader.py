import os
from torch.utils.data import Dataset,ConcatDataset,DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image, ImageFile
from .syszux_log import LOG
import cv2

#Dataset -> VisionDataset -> DatasetFolder -> ImageFolder -> *Dataset

class ImageFolderWithTransformDataset(ImageFolder):
    def __init__(self, deepvac_config):
        self.transform_op = deepvac_config.transform_op
        self.img_folder = deepvac_config.img_folder
        super(ImageFolderWithTransformDataset,self).__init__(self.img_folder, self.transform_op)

class ImageFolderWithPathsDataset(ImageFolderWithTransformDataset):
    def __init__(self, deepvac_config):
        super(ImageFolderWithPathsDataset, self).__init__(deepvac_config)
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPathsDataset, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        #print("gemfieldimg: ", tuple_with_path[0],'\t',tuple_with_path[1],'\t',tuple_with_path[2])
        return tuple_with_path

class FileLineDataset(Dataset):
    def __init__(self, deepvac_config):
        self.path_prefix = deepvac_config.fileline_data_path_prefix
        self.fileline_path = deepvac_config.fileline_path
        self.transform = deepvac_config.transform
        self.samples = []
        mark = []

        with open(self.fileline_path) as f:
            for line in f:
                label = self._buildLabelFromLine(line)
                self.samples.append(label)
                mark.append(label[1])

        self.len = len(self.samples)
        self.class_num = len(np.unique(mark))
        LOG.logI('FileLineDataset size: {} / {}'.format(self.len, self.class_num))

    def _buildLabelFromLine(self, line):
        line = line.strip().split(" ")
        return [line[0], int(line[1])]

    def __getitem__(self, index):
        path, target = self.samples[index]
        abs_path = os.path.join(self.path_prefix, path)
        return self._buildSampleFromPath(abs_path), target

    def _buildSampleFromPath(self, abs_path):
        #we just set default loader with Pillow Image
        sample = Image.open(abs_path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len

class FileLineCvStrDataset(FileLineDataset):
    def _buildLabelFromLine(self, line):
        line = line.strip().split(" ", 1)
        return [line[0], line[1]]

    def _buildSampleFromPath(self, abs_path):
        #we just set default loader with Pillow Image
        sample = cv2.imread(abs_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class OsWalkerLoader(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.input_dir = self.conf.input_dir

    def __call__(self, input_dir=None):
        if input_dir:
            self.input_dir = input_dir

        files = []
        for subdir, dirs, fns in os.walk(self.input_dir):
            for fn in fns:
                filepath = os.path.join(subdir, fn)
                files.append(filepath)
        return files


class CocoCVDataset(Dataset):
    def __init__(self, deepvac_config):
        super(CocoCVDataset, self).__init__()
        self.conf = deepvac_config
        self.transform = deepvac_config.transform
        self.img_folder = deepvac_config.img_folder
        self.aug_executor = deepvac_config.aug_executor

        try:
            from pycocotools.coco import COCO
        except:
            raise Exception("pycocotools module not found, you should try 'pip3 install pycocotools' first!")
        self.coco = COCO(deepvac_config.annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cats = list(sorted(self.coco.cats.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample, label = self._getSample(index)
        # augment for numpy
        if self.aug_executor:
            sample = self.aug_executor(sample)
        # pytorch offical transform for PIL Image or numpy
        if self.transform is not None:
            # rewrite method: '_buildSample' first
            sample = self.transform(sample)
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
        img = cv2.imread(os.path.join(self.img_folder, file_name), 1)
        assert img is not None, "Image {} not found!".format(file_name)
        return img

    def loadAnns(self, index):
        ann_ids = self.coco.getAnnIds(imgIds=self.ids[index])
        anns = self.coco.loadAnns(ann_ids)
        category_ids = np.array([self.cats.index(i["category_id"]) for i in anns], dtype=np.float)
        boxes = np.array([i["bbox"] for i in anns], dtype=np.float)
        masks = np.array([self.coco.annToMask(i) for i in anns], dtype=np.float)
        return category_ids, boxes, masks
