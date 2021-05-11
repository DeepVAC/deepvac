import os
import cv2
from torch.utils.data import Dataset

class OsWalkDataset(Dataset):
    def __init__(self, deepvac_config):
        super(OsWalkDataset, self).__init__()
        self.transform = deepvac_config.transform
        self.files = []
        for subdir, dirs, fns in os.walk(deepvac_config.input_dir):
            for fn in fns:
                self.files.append(os.path.join(subdir, fn))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath = self.files[index]
        sample = cv2.imread(filepath)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample