from torchvision.datasets import ImageFolder

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