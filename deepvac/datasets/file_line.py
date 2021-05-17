import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from ..utils import LOG

class FileLineDataset(Dataset):
    def __init__(self, deepvac_config, fileline_path, delimiter=' ', sample_path_prefix=''):
        self.config = deepvac_config.datasets
        self.transform = self.config.transform
        self.composer = self.config.composer
        self.sample_path_prefix = sample_path_prefix
        self.fileline_path = fileline_path
        self.delimiter = delimiter
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
        line = line.strip().split(self.delimiter)
        return [line[0], int(line[1])]

    def __getitem__(self, index):
        path, target = self.samples[index]
        abs_path = os.path.join(self.sample_path_prefix, path)
        return self._buildSampleFromPath(abs_path), target

    def _buildSampleFromPath(self, abs_path):
        #we just set default loader with Pillow Image
        sample = Image.open(abs_path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.composer is not None:
            sample = self.composer(sample)
        return sample

    def __len__(self):
        return self.len

class FileLineCvStrDataset(FileLineDataset):
    def _buildLabelFromLine(self, line):
        line = line.strip().split(self.delimiter, 1)
        return [line[0], line[1]]

    def _buildSampleFromPath(self, abs_path):
        #we just set default loader with Pillow Image
        sample = cv2.imread(abs_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample