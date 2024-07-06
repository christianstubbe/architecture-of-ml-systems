import os
import torch
import rasterio
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from tqdm.notebook import tqdm 
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# from tqdm import tqdm


class CityDataset(Dataset):
    def __init__(self, 
                 path, # path to the data. Should contain directories of each city with the data
                 patch_size=32, # size of the patches (patch_size x patch_size)
                 transform=None, # transform to apply to the data
                 data_name="openEO.tif", # name of the data file (must be consistent in all directories)
                 labels_name="building_mask_dense.tif", # name of the labels file (must be consistent in all directories)
                 scale_data=True, # if True the data is scaled to 0-1
                 image_bands=[1,2,3,4,5, 6], # bands to load
                 stride=None, # if None stride is equal to patch_size
                 min_labels:float=0.1, # between 0.0 and 1.0, minimum percentage of labels in a patch
                 devrun=False, # if True only load data from the first two cities
                 ):
        self.devrun=devrun
        self.path = path
        self.transform = transform
        self.image_bands = image_bands
        self.scale_data = scale_data
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size
        self.min_labels = min_labels # minimum percentage of labels in a patch

        # get data and labels
        self.data, self.labels = self.__load_data(data_name, labels_name)

        # create patches
        self.X, self.y = self.__create_image_patches()


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        item = {"data": self.X[idx], "labels": self.y[idx]}
        if self.transform:
            item = self.transform(item)
        return item
    

    def update_patch_size(self, patch_size):
        self.patch_size = patch_size
        self.X, self.y = self.__create_image_patches()


    def train_val_split(self, val_size=0.2, n_groups=100, random_state=42):
        """
        Test Val split, returns two dataloaders that contain the test and validation data
        the test and validation data should have patches with equally distributed labels
        
        n_groups: int, number of groups to create for stratification 1 would mean random split

        We need to stratify the data to create a test and validation set that have equally distributed labels
        However some labels only appear once. Therfore we need to create groups of labels that have the same percentage of labels.
        We can then stratify the data based on these groups.
        This works!

        """

        # create groups of labels (n_groups groups with equal percentage of samples)
        y_sums = [np.sum(y) for y in self.y]
        sorted_indices = np.argsort(y_sums)
        group_size = len(self.y) // n_groups
        groups = np.zeros(len(self.y), dtype=int)
        for i in range(n_groups):
            start_idx = i*group_size
            end_idx = (i+1)*group_size if i < n_groups-1 else len(self.y)
            groups[sorted_indices[start_idx:end_idx]] = i


        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        for train_indices, val_indices in stratified_split.split(self.X, groups):
            train_idxs = train_indices
            val_idxs = val_indices
        
        train_ds = Subset(self, train_idxs)
        val_ds = Subset(self, val_idxs)
        return train_ds, val_ds




            
    def __create_image_patches(self):
        # self.__create_image_patches_from_image(self.data[0], self.labels[0])
        assert len(self.data) == len(self.labels), "Data and Labels don't match"
        assert all([d.shape[1] == l.shape[0] and d.shape[2] == l.shape[1] for d, l in zip(self.data, self.labels)]), "Data and Labels don't match"

        X = []
        y = []
        for (d, l) in tqdm(zip(self.data, self.labels), desc="Creating Patches from Images"):
            x, yy = self.__create_image_patches_from_image(d, l)
            X.extend(x)
            y.extend(yy)
        return X, y
        

    def __create_image_patches_from_image(self, image, label_image):
        # print(image.shape)
        width = image.shape[1]
        height = image.shape[2]
        patches = []
        labels = []
        # first check if how many pixels are left in the last patch in each dimension
        left_width = width % self.stride
        left_height = height % self.stride
        # if there are pixels left in the last patch we center the patches, so that the left pixels are distributed equally
        start_width = left_width // 2
        start_height = left_height // 2
        # iterate over the image and create patches
        for i in range(start_width, (width - self.patch_size - start_width + 1), self.stride):
            for j in range(start_height, (height - self.patch_size - start_height + 1), self.stride):

                # check if the patch has enough labels
                label_patch = label_image[i:i+self.patch_size, j:j+self.patch_size]
                if not self.__check_good_labels_ratio(label_patch):
                    continue

                patch = image[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
                labels.append(label_patch)
        # print(len(patches), len(labels))
        return patches, labels



    def __check_good_labels_ratio(self, label_patch):
        return np.sum(label_patch) / (self.patch_size ** 2) >= self.min_labels
        
    def __load_image_data(self, path):
        with rasterio.open(os.path.join(path)) as src:
            data = src.read(self.image_bands)
        if self.scale_data:
            data = self.__scaler(data, 1, 99)
        return data
    
    def __load_data(self, data_name, labels_name): 
        directories = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and not d.startswith(".")]
        # check if ther is a file with name data_name in each directory
        assert all([data_name in os.listdir(os.path.join(self.path, d)) for d in directories]), f"Data file {data_name} not found in all directories"
        assert all([labels_name in os.listdir(os.path.join(self.path, d)) for d in directories]), f"Labels file {labels_name} not found in all directories"
        data_city_paths = [os.path.join(self.path, d, data_name) for d in directories]
        labels_city_paths = [os.path.join(self.path, d, labels_name) for d in directories]
        print("loading data from cities:")
        print([d.split("/")[-2] for d in data_city_paths])

        if self.devrun:
            data_city_paths = data_city_paths[:2]
            labels_city_paths = labels_city_paths[:2]

        return [self.__load_image_data(p) for p in tqdm(data_city_paths, desc="Loading Images")], [self.__load_label_data(p) for p in tqdm(labels_city_paths, desc="Loading Labels")]
    
    @staticmethod
    def __load_label_data(path):
        with rasterio.open(os.path.join(path)) as src:
            data = src.read(1)
        return data

    @staticmethod
    def __scaler(bands, p_low=0.5, p_high=99.5):
        p_down, p_up = np.percentile(bands, (p_low, p_high), axis=(1, 2))
        return np.clip((bands - p_down[:, None, None]) / (p_up - p_down)[:, None, None], 0, 1).astype(np.float32)

            


if __name__ == "__main__":
    dataset = CityDataset("/home/jlb/Projects/architecture-of-ml-systems/data/train", devrun=True)
    print(len(dataset))
    print(dataset[0]["data"].shape, dataset[0]["labels"].shape)
    train_dataset, val_dataset = dataset.train_val_split()
    print("len train_dataset",len(train_dataset))
    dl_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("len dl_train",len(dl_train))
    print("len val_dataset",len(val_dataset))
    dl_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("len dl_val",len(dl_val))
    sample1 = next(iter(dl_train))
    print(sample1["data"].shape, sample1["labels"].shape)
    sample2 = next(iter(dl_val))
    print(sample2["data"].shape, sample2["labels"].shape)
