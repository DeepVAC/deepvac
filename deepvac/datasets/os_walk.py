import os
import cv2
from .base_dataset import DatasetBase

class OsWalkDataset(DatasetBase):
    def __init__(self, deepvac_config, sample_path):
        super(OsWalkDataset, self).__init__(deepvac_config)
        self.files = []
        for subdir, dirs, fns in os.walk(sample_path):
            for fn in fns:
                self.files.append(os.path.join(subdir, fn))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.files[index]
        sample = cv2.imread(filepath)
        if self.config.transform is not None:
            sample = self.config.transform(sample)
        if self.config.composer is not None:
            sample = self.config.composer(sample)
        return sample