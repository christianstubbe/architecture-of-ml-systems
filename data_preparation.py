import rasterio
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from utils import stretch_hist
import matplotlib.pyplot as plt


class DataPrepare:
    def __init__(self, datahandler, logger, patch_size=128):
        """
        
        """
        self.logger = logger
        self.patch_size = patch_size
        self.dataHandler = datahandler
    

    def create_tensor_of_windows(self, city):
        """
        Create tensor with dimensions [N, H, W, C] from the satellite image of the city.
        """
        # Load image and Mask
        image = self.dataHandler.get_satellite_image(city)
        mask = self.dataHandler.get_building_mask(city)

        # Merge Mask onto Image
        image_with_mask = np.dstack((image, mask))

        # cut of edges so image shape is divisible by patch size
        reduced_image = image_with_mask[:-(image_with_mask.shape[0]%self.patch_size), :-(image_with_mask.shape[1]%self.patch_size)]


        # calculate number of patches
        N = reduced_image.shape[0]//self.patch_size*reduced_image.shape[1]//self.patch_size

        # initialize target array
        target_array = np.zeros((N, self.patch_size, self.patch_size, reduced_image.shape[-1]), dtype=np.uint16)

        # fill target array
        for row in range(self.patch_size):
            for col in range(self.patch_size):
                # calculate row and column indices
                row_filter = range(row,reduced_image.shape[0]+row,self.patch_size)
                col_filter = range(col,reduced_image.shape[1]+col,self.patch_size)

                # write values into target array
                target_array[:, row, col, :] = reduced_image[row_filter][:,col_filter,:].reshape(-1, reduced_image.shape[-1])

        return target_array


   
def divide_into_test_training(data, train_ratio=0.8):
    """
    Divide the data into test and training split.
    """
    
    # Define the split ratio
    test_ratio = 1 - train_ratio

    # Calculate the sizes for training and test sets
    train_size = int(train_ratio * len(data))
    test_size = len(data) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size = 64):
    """
    Create DataLoaders.
    """
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_data_loaders(data, train_ratio = 0.8, batch_size = 64):
    train_ds, test_ds = divide_into_test_training(data,train_ratio=train_ratio)
    train_loader, test_loader = create_data_loaders(train_ds, test_ds, batch_size=batch_size)
    return train_loader, test_loader

def plot_sub_image( image_data):
    """
    Plot sub image.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(stretch_hist(image_data[:,:,:3]))
    ax[1].imshow(stretch_hist(image_data[:,:,-1]))
    return fig
