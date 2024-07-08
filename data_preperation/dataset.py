import os
import torch
import rasterio
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd

# from tqdm import tqdm


class CityDataset(Dataset):
    """
    A custom dataset class for handling city-based data.

    Args:
        path (str, optional): Path to the data. Should contain directories of each city with the data.
        patch_size (int, optional): Size of the patches (patch_size x patch_size).
        transform (callable, optional): Transform to apply to the data.
        data_name (str, optional): Name of the data file (must be consistent in all directories).
        labels_name (str, optional): Name of the labels file (must be consistent in all directories).
        scale_data (bool, optional): If True, the data is scaled to the range of 0-1.
        image_bands (list, optional): Bands to load.
        stride (int, optional): If provided, the stride between patches. If not provided, it will be set to the patch size.
        min_labels (float, optional): Minimum percentage of labels in a patch (between 0.0 and 1.0).
        devrun (bool, optional): If True, only load data from the first two cities.
        cities (list, optional): If provided, only load data from the specified cities.
        train (bool, optional): If True, the dataset is used for training. If False, the dataset is used for testing.
        images (list, optional): List of images.
        buildings_mask (ndarray, optional): Array of building masks.
        boundaries_mask (ndarray, optional): Array of boundaries masks.

    Attributes:
        __devrun (bool): Flag indicating if devrun mode is enabled.
        __path (str): Path to the data.
        transform (callable): Transform to apply to the data.
        __image_bands (list): Bands to load.
        __scale_data (bool): Flag indicating if the data should be scaled.
        __patch_size (int): Size of the patches.
        __stride (int): Stride between patches.
        __min_labels (float): Minimum percentage of labels in a patch.
        __cities (list): List of cities.
        __boundaries_mask (ndarray): Array of boundaries masks.
        train (bool): Flag indicating if the dataset is used for training.
        data (list): List of data.
        labels (ndarray): Array of labels.
        X (list): List of image patches.
        y (list): List of label patches.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the specified index.
        get_cities(): Returns the list of cities in the dataset.
        set_channels(channels): Set the number of image channels for the dataset.
        get_image_bands(): Returns the image bands of the dataset.
        update_patch_size(patch_size, stride): Updates the patch size and stride of the dataset.
        train_val_split(val_size, n_groups, random_state, show_summary): Splits the dataset into training and validation sets.
        __print_summary(train_ds, val_ds, train_indices, val_indices): Prints a summary of the dataset split.
        __get_city_by_index(idx): Returns the city corresponding to the given index.
        __create_image_patches(): Creates image patches from the data.
        __create_image_patches_from_image(image, label_image, idx): Creates image patches from a single image.

    """

    def __init__(
        self,
        path=None,
        patch_size=32,
        transform=None,
        data_name="openEO.tif",
        labels_name="building_mask_dense.tif",
        scale_data=True,
        image_bands=[1, 2, 3, 4, 5, 6],
        stride=None,
        min_labels: float = 0.1,
        devrun=False,
        cities=None,
        train=True,
        images=None,
        buildings_mask=None,
        boundaries_mask=None,
    ):
        self.__devrun = devrun
        self.__path = path
        self.transform = transform
        self.__image_bands = image_bands
        self.__scale_data = scale_data
        self.__patch_size = patch_size
        self.__stride = stride if stride else patch_size
        self.__min_labels = min_labels
        self.__cities = cities
        self.__boundaries_mask = boundaries_mask
        self.train = train

        if (
            images is not None
            and buildings_mask is not None
            and boundaries_mask is not None
        ):
            self.data = [self.__scaler(self.__fill_na(img), 1, 99) for img in images]
            self.labels = buildings_mask
            self.boundaries_mask = boundaries_mask
            self.__image_bands = list(range(images[0].shape[0]))
            self.X, self.y = self.__create_image_patches()
        elif self.__path is not None:
            # get data and labels
            self.data, self.labels = self.__load_data(data_name, labels_name)
            self.__image_bands = list(range(self.data[0].shape[0]))
        else:
            raise ValueError(
                "Either path or (data, labels and boundaries_mask) must be provided"
            )

        # create patches
        if self.train:
            self.X, self.y = self.__create_image_patches()
        else:
            self.X = self.data
            self.y = self.labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            "data": self.X[idx][self.__image_bands],
            "labels": np.array([self.y[idx]]),
        }
        if self.transform:
            item = self.transform(item)
        return item

    def get_cities(self):
        """
        Returns the list of cities in the dataset.

        Returns:
            list: A list of cities.
        """
        return self.__cities

    def set_channels(self, channels):
        """
        Set the number of image channels for the dataset.

        Parameters:
        - channels (int): The number of image channels.

        Returns:
        None
        """
        self.__image_bands = channels

    def get_image_bands(self):
        """
        Returns the image bands of the dataset.

        Returns:
            list: A list of image bands.
        """
        return self.__image_bands

    def update_patch_size(self, patch_size: int, stride: int = None):
        """
        Updates the patch size and stride of the dataset.

        Parameters:
        - patch_size (int): The size of the patches.
        - stride (int, optional): The stride between patches. If not provided, it will be set to the patch size.

        Returns:
        None
        """
        self.__patch_size = patch_size
        self.__stride = stride if stride else patch_size
        self.X, self.y = self.__create_image_patches()

    def train_val_split(
        self, val_size=0.2, n_groups=100, random_state=42, show_summary=True
    ):
        """
        Splits the dataset into training and validation sets.

        Parameters:
        - val_size (float, optional): The size of the validation set (default: 0.2).
        - n_groups (int, optional): The number of groups to create for stratified splitting (default: 100).
        - random_state (int, optional): The random state for reproducibility (default: 42).
        - show_summary (bool, optional): If True, print a summary of the dataset split (default: True).

        Returns:
        - train_ds (Subset): The training dataset.
        - val_ds (Subset): The validation dataset.
        """
        # create groups of labels (n_groups groups with equal percentage of samples)
        y_sums = [np.sum(y) for y in self.y]
        sorted_indices = np.argsort(y_sums)
        group_size = len(self.y) // n_groups
        groups = np.zeros(len(self.y), dtype=int)
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < n_groups - 1 else len(self.y)
            groups[sorted_indices[start_idx:end_idx]] = i

        stratified_split = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=random_state
        )
        for train_indices, val_indices in stratified_split.split(self.X, groups):
            train_idxs = train_indices
            val_idxs = val_indices

        train_ds = Subset(self, train_idxs)
        val_ds = Subset(self, val_idxs)

        if show_summary:
            # print out summary stats of each split
            self.__print_summary(train_ds, val_ds, train_indices, val_indices)

        return train_ds, val_ds

    def __print_summary(self, train_ds, val_ds, train_indices, val_indices):
        print("Train:")
        print("Number of samples:", len(train_ds))
        print(
            "Shape of Train data (data, label)",
            train_ds[0]["data"].shape,
            train_ds[0]["labels"].shape,
        )
        print("Val:")
        print("Number of samples:", len(val_ds))
        print(
            "Shape of Val data (data, label)",
            val_ds[0]["data"].shape,
            val_ds[0]["labels"].shape,
        )
        # get the mean percentage of labels
        print("*" * 50)
        print(
            "Mean percentage of 1 labels in train:",
            np.mean([np.sum(d["labels"]) / (self.__patch_size**2) for d in train_ds]),
        )
        print(
            "Mean percentage of 1 labels in val:",
            np.mean([np.sum(d["labels"]) / (self.__patch_size**2) for d in val_ds]),
        )
        print(
            "Mean percentage of 1 labels in all data:",
            np.mean([np.sum(d["labels"]) / (self.__patch_size**2) for d in self]),
        )
        print("*" * 50)
        print(
            f"Std of percentage 1 labels in train: {np.std([np.sum(d['labels']) / (self.__patch_size ** 2) for d in train_ds])}"
        )
        print(
            f"Std of percentage 1 labels in val: {np.std([np.sum(d['labels']) / (self.__patch_size ** 2) for d in val_ds])}"
        )
        print(
            f"Std of percentage 1 labels in all data: {np.std([np.sum(d['labels']) / (self.__patch_size ** 2) for d in self])}"
        )
        print("*" * 50)
        print(
            f"Min percentage of 1 labels in train: {np.min([np.sum(d['labels']) / (self.__patch_size ** 2) for d in train_ds])}"
        )
        print(
            f"Min percentage of 1 labels in val: {np.min([np.sum(d['labels']) / (self.__patch_size ** 2) for d in val_ds])}"
        )
        print(
            f"Min percentage of 1 labels in all data: {np.min([np.sum(d['labels']) / (self.__patch_size ** 2) for d in self])}"
        )
        print("*" * 50)
        print(
            f"Max percentage of 1 labels in train: {np.max([np.sum(d['labels']) / (self.__patch_size ** 2) for d in train_ds])}"
        )
        print(
            f"Max percentage of 1 labels in val: {np.max([np.sum(d['labels']) / (self.__patch_size ** 2) for d in val_ds])}"
        )
        print(
            f"Max percentage of 1 labels in all data: {np.max([np.sum(d['labels']) / (self.__patch_size ** 2) for d in self])}"
        )
        print("*" * 50)
        train_cities = self.__get_city_by_index(train_indices)
        val_cities = self.__get_city_by_index(val_indices)
        # lookup how often each city occured in the different sets
        train_city_counts = np.unique(train_cities, return_counts=True)
        validation_citiy_counts = np.unique(val_cities, return_counts=True)

        # create dataframes for better readability
        df = pd.DataFrame(
            {
                "Train": pd.Series(
                    train_city_counts[1] / train_cities.shape[0],
                    index=train_city_counts[0],
                    name="train",
                ),
                "Validation": pd.Series(
                    validation_citiy_counts[1] / val_cities.shape[0],
                    index=validation_citiy_counts[0],
                    name="validation",
                ),
            }
        )

        df.index = df.index.map({i: c for i, c in enumerate(self.__cities)})
        print(
            "Comparison of cities the data in the differen sets originates from:\n",
            df.T,
        )

    def __get_city_by_index(self, idx):
        return self.__cities_indices[idx]

    def __create_image_patches(self):
        # self.__create_image_patches_from_image(self.data[0], self.labels[0])
        assert len(self.data) == len(self.labels), "Data and Labels don't match"
        assert all(
            [
                d.shape[1] == l.shape[0] and d.shape[2] == l.shape[1]
                for d, l in zip(self.data, self.labels)
            ]
        ), "Data and Labels don't match"
        self.__cities_indices = []
        X = []
        y = []
        for i, (d, l) in tqdm(
            enumerate(zip(self.data, self.labels)), desc="Creating Patches from Images"
        ):
            x, yy = self.__create_image_patches_from_image(d, l, i)
            self.__cities_indices.append(np.ones(len(x), dtype=np.uint8) * i)
            X.extend(x)
            y.extend(yy)
        self.__cities_indices = np.concatenate(self.__cities_indices)
        return X, y

    def __create_image_patches_from_image(self, image, label_image, idx):
        # print(image.shape)
        width = image.shape[1]
        height = image.shape[2]
        patches = []
        labels = []
        # first check if how many pixels are left in the last patch in each dimension
        left_width = width % self.__stride
        left_height = height % self.__stride
        # if there are pixels left in the last patch we center the patches, so that the left pixels are distributed equally
        start_width = left_width // 2
        start_height = left_height // 2
        # iterate over the image and create patches
        for i in range(
            start_width, (width - self.__patch_size - start_width + 1), self.__stride
        ):
            for j in range(
                start_height,
                (height - self.__patch_size - start_height + 1),
                self.__stride,
            ):

                #  check if within the osm boundaries
                if self.__boundaries_mask is not None:
                    boundaries_patch = self.__boundaries_mask[idx][
                        i : i + self.__patch_size, j : j + self.__patch_size
                    ]
                    if any(boundaries_patch.flatten() < 1):
                        continue
                # check if the patch has enough labels
                label_patch = label_image[
                    i : i + self.__patch_size, j : j + self.__patch_size
                ]
                if not self.__check_good_labels_ratio(label_patch):
                    continue

                patch = image[:, i : i + self.__patch_size, j : j + self.__patch_size]
                patches.append(patch)
                labels.append(label_patch)
        # print(len(patches), len(labels))
        return patches, labels

    def __check_good_labels_ratio(self, label_patch):
        return np.sum(label_patch) / (self.__patch_size**2) >= self.__min_labels

    def __load_image_data(self, path):
        """
        Loads image data fromm disc, filters out negative values and scales the data if needed
        """
        with rasterio.open(os.path.join(path)) as src:
            data = src.read(self.__image_bands)

        # filter out negaitve vals
        data = self.__fill_na(data)

        if self.__scale_data:
            data = self.__scaler(data, 1, 99)
        return data

    def __load_data(self, data_name, labels_name):
        directories = [
            d
            for d in os.listdir(self.__path)
            if os.path.isdir(os.path.join(self.__path, d)) and not d.startswith(".")
        ]

        if self.__cities:
            directories = [d for d in directories if d in self.__cities]

        # check if there is a file with name data_name in each directory
        assert all(
            [data_name in os.listdir(os.path.join(self.__path, d)) for d in directories]
        ), f"Data file {data_name} not found in all directories"
        assert all(
            [
                labels_name in os.listdir(os.path.join(self.__path, d))
                for d in directories
            ]
        ), f"Labels file {labels_name} not found in all directories"
        data_city_paths = [os.path.join(self.__path, d, data_name) for d in directories]
        labels_city_paths = [
            os.path.join(self.__path, d, labels_name) for d in directories
        ]

        print("Loading data from cities:")
        print([d.split("/")[-2] for d in data_city_paths])

        if self.__devrun:
            data_city_paths = data_city_paths[:2]
            labels_city_paths = labels_city_paths[:2]

        return [
            self.__load_image_data(p)
            for p in tqdm(data_city_paths, desc="Loading Images")
        ], [
            self.__load_label_data(p)
            for p in tqdm(labels_city_paths, desc="Loading Labels")
        ]

    @staticmethod
    def __load_label_data(path):
        with rasterio.open(os.path.join(path)) as src:
            data = src.read(1)
        return data

    @staticmethod
    def __scaler(bands, p_low=0.5, p_high=99.5):
        p_down, p_up = np.percentile(bands, (p_low, p_high), axis=(1, 2))
        return np.clip(
            (bands - p_down[:, None, None]) / (p_up - p_down)[:, None, None], 0, 1
        ).astype(np.float32)

    @staticmethod
    def __fill_na(data):
        data[data < 0] = 0
        return data


if __name__ == "__main__":
    print("Testing CityDataset")
    dataset = CityDataset(
        "/home/jlb/Projects/architecture-of-ml-systems/data/train", devrun=True
    )
    print(len(dataset))
    print(dataset[0]["data"].shape, dataset[0]["labels"].shape)
    train_dataset, val_dataset = dataset.train_val_split()
    print("len train_dataset", len(train_dataset))
    dl_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("len dl_train", len(dl_train))
    print("len val_dataset", len(val_dataset))
    dl_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("len dl_val", len(dl_val))
    sample1 = next(iter(dl_train))
    print(sample1["data"].shape, sample1["labels"].shape)
    sample2 = next(iter(dl_val))

    print(sample2["data"].shape, sample2["labels"].shape)

    print("TESTING UPDATE PATCH SIZE")
    dataset.update_patch_size(64)
    print(len(dataset))
