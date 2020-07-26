import os
from torch.utils.data import Dataset,ConcatDataset,DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

#Dataset -> VisionDataset -> DatasetFolder -> ImageFolder -> *DatasetLoader

class ImageFolderWithTransformDatasetLoader(ImageFolder):
    def __init__(self, deepvac_config):
        self.transform_op = deepvac_config.loader.transform_op
        self.img_folder = deepvac_config.loader.img_folder
        super(ImageFolderWithTransformDatasetLoader,self).__init__(self.img_folder, self.transform_op)

class ImageFolderWithPathsDatasetLoader(ImageFolderWithTransformDatasetLoader):
    def __init__(self, deepvac_config):
        super(ImageFolderWithPathsDatasetLoader, self).__init__(deepvac_config)
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPathsDatasetLoader, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        #print("gemfieldimg: ", tuple_with_path[0],'\t',tuple_with_path[1],'\t',tuple_with_path[2])
        return tuple_with_path


class FileLineDatasetLoader(Dataset):
    def __init__(self, deepvac_config):
        self.path_prefix = deepvac_config.loader.path_prefix
        self.fileline_path = deepvac_config.loader.fileline_path
        self.transforms = deepvac_config.loader.transforms
        self.samples = []

        with open(self.fileline_path) as f:
            for line in f:
                line = line.strip().split(" ")
                label = [line[0], int(line[1])]
                self.samples.append(label)

        self.len = len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        abs_path = os.path.join(self.path_prefix, path)
        #we just set default loader with Pillow Image
        sample = Image.open(abs_path).convert('RGB')
        if self.transform is not None:
            sample = self.transforms(sample)

        return sample, target

    def __len__(self):
        return self.len

class OsWalkerLoader(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        self.input_dir = self.conf.input_dir

    def __call__(self):
        for subdir, dirs, files in os.walk(self.input_dir):
            for file in files:
                #print os.path.join(subdir, file)
                filepath = subdir + os.sep + file
                yield filepath
    
# class AuditBase(object):
#     def __init__(self, deepvac_config):
#         self.conf = deepvac_config

#     def setMaterialsInput(self, input_path):
#         self.input_path = input_path
#         self.file_loader = self.getFileList(self.input_path)
#         return self

#     def getFileList(self, target_dir):
#         for subdir, dirs, files in os.walk(target_dir):
#             for file in files:
#                 #print os.path.join(subdir, file)
#                 filepath = subdir + os.sep + file
#                 ground_truth = subdir.split(os.sep)[-1]
#                 yield filepath

#     def process(self,f):
#         LOG.logI("Just print {}".format(f))

#     def __call__(self):
#         for f in self.file_loader:
#             self.process(f)