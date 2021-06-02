import os
import cv2
from .base_dataset import DatasetBase

class OsWalkBaseDataset(DatasetBase):
    def __init__(self, deepvac_config, sample_path):
        super(OsWalkBaseDataset, self).__init__(deepvac_config)
        self.files = []
        for subdir, dirs, fns in os.walk(sample_path):
            for fn in fns:
                self.files.append(os.path.join(subdir, fn))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.files[index]
        sample = cv2.imread(filepath)
        sample = self.compose(sample)
        return sample, filepath

class OsWalkDataset(OsWalkBaseDataset):
    def __getitem__(self, index):
        sample, filepath = super(OsWalkDataset, self).__getitem__(index)
        return sample

class OsWalkContinueDataset(OsWalkBaseDataset):
    def __getitem__(self, index):
        try:
            sample, filepath = super(OsWalkDataset, self).__getitem__(index)
        except:
            sample, filepath=None,None
        return sample

