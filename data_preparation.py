import rasterio
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from utils import stretch_hist
import matplotlib.pyplot as plt
from scipy.special import kl_div
import pandas as pd


def create_tensor_of_windows(image, mask, patch_size=128):
    """
    Create tensor with dimensions [N, H, W, C+1] from the satellite image of the city.
    image should be of shape (H, W, C)
    mask should be of shape (H, W, 1)
    """
    # Merge Mask onto Image
    image_with_mask = np.dstack((image, mask))

    # cut of edges so image shape is divisible by patch size
    reduced_image = image_with_mask[:-(image_with_mask.shape[0]%patch_size), :-(image_with_mask.shape[1]%patch_size)]

    # calculate number of patches
    N = reduced_image.shape[0]//patch_size*reduced_image.shape[1]//patch_size

    # initialize target array
    target_array = np.zeros((N, patch_size, patch_size, reduced_image.shape[-1]), dtype=np.uint16)

    # fill target array
    for row in range(patch_size):
        for col in range(patch_size):
            # calculate row and column indices
            row_filter = range(row,reduced_image.shape[0]+row,patch_size)
            col_filter = range(col,reduced_image.shape[1]+col,patch_size)

            # write values into target array
            target_array[:, row, col, :] = reduced_image[row_filter][:,col_filter,:].reshape(-1, reduced_image.shape[-1])

    return target_array

   
def divide_into_test_training(data, test_ratio=0.2, validation_ratio=0, seed=42):
    """
    Divide the data into test and training split with seed.
    """
    
    # Define the split rati
    train_ratio = 1 - test_ratio - validation_ratio
    if train_ratio < 0:
        raise ValueError("The train ratio is negative. Please check the split ratios.")

    # Calculate the sizes for training and test sets
    train_size = int(train_ratio * len(data))
    test_size = int(test_ratio * len(data))
    validation_size = int(validation_ratio * len(data))

    # Split the dataset with seed
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset, validation_dataset = random_split(data, [train_size, test_size, validation_size], generator=generator)

    return train_dataset, test_dataset, validation_dataset


def validate_test_training_validation_split(train_dataset, test_dataset, validation_dataset, city_names=None):
    """
    Validate the train to the test split and the train to the validation split.
    """
    kl_div_train_test = kl_div(train_dataset, test_dataset)
    print(kl_div_train_test)
    
    kl_div_train_validation = kl_div(train_dataset, validation_dataset)
    print(kl_div_train_validation)

    if city_names is not None:  
        train_cities = train_dataset.dataset[:,-1,0,0][train_dataset.indices]
        test_cities = test_dataset.dataset[:,-1,0,0][test_dataset.indices]
        validation_cities = validation_dataset.dataset[:,-1,0,0][validation_dataset.indices]
        train_city_counts = np.unique(train_cities, return_counts=True)
        test_city_counts = np.unique(test_cities, return_counts=True)
        validation_citiy_counts = np.unique(validation_cities, return_counts=True)

        df = pd.DataFrame({
            "train":pd.Series(train_city_counts[1]/train_cities.shape[0], index=train_city_counts[0], name='train'),
            "test":pd.Series(test_city_counts[1]/test_cities.shape[0], index=test_city_counts[0], name='test'),
            "validation":pd.Series(validation_citiy_counts[1]/validation_cities.shape[0], index=validation_citiy_counts[0], name='validation')})

        df.index = city_names
        print(df)




def create_data_loaders(train_dataset, test_dataset, batch_size = 64):
    """
    Create DataLoaders.
    """
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def apply_preprocessing_pipeline(images, masks, patch_size = 128, test_ratio = 0.2,validation_ratio=0, batch_size = 64, cities=None):
    """
    applies windowing, deviding into train and test and creating data loaders.
    """

    # for each city create patched images
    patched_images = []
    for i,(image, mask ) in enumerate(zip(images, masks)):
        patched_image = create_tensor_of_windows(image, mask, patch_size=patch_size)
        city = np.ones(shape=list(patched_image.shape[:-1])+[1])*i
        patched_image_with_city = np.concatenate([patched_image, city], axis=-1)
        patched_images.append(patched_image_with_city)


    # concatenate all patched images
    patched_images_merged = np.concatenate(patched_images, axis=0)

    # reorder axis to [N, C, H, W] for torch
    patched_images_merged = np.transpose(patched_images_merged, (0,3,1,2))

    # devide into train and test
    train_ds, test_ds, val_ds = divide_into_test_training(patched_images_merged, test_ratio=test_ratio, validation_ratio=validation_ratio)

    validate_test_training_validation_split(train_ds, test_ds, val_ds)

    # create data loaders
    train_loader, test_loader , validation_loader= create_data_loaders(train_ds, test_ds,val_ds, batch_size=batch_size)

    return train_loader, test_loader, validation_loader


def plot_sub_image( image_data):
    """
    Plot sub image.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(stretch_hist(image_data[:,:,:3]))
    ax[1].imshow(stretch_hist(image_data[:,:,-1]))
    return fig
