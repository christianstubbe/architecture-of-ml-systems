import rasterio
import numpy as np
import torch
from torch.utils.data import random_split

class DataPrepare:
    def __init__(self, DataHandler, logger):
        """
        
        """
        self.logger = logger
        self.patch_size = 256
        self.dataHandler = DataHandler(logger)
    

    def create_tensor(self, city):
        """
        Create tensor with dimensions [N, H, W, C] from the satellite image of the city.
        """
        image = self.DataHandler.get_satellite_image(city)

        # 
        image = np.swapaxes(image, 0,1)
        image = np.swapaxes(image, 1,2)

        # cut of edges so image shape is divisible by patch size
        reduced_image = image[:-(image.shape[0]%self.patch_size), :-(image.shape[1]%self.patch_size)]


        # calculate number of patches
        N = reduced_image.shape[0]//self.patch_size*reduced_image.shape[1]//self.patch_size

        # initialize target array
        target_array = np.zeros((N, self.patch_size, self.patch_size, reduced_image.shape[-1]))

        # fill target array
        for row in range(self.patch_size):
            for col in range(self.patch_size):
                # calculate row and column indices
                row_filter = range(row,reduced_image.shape[0]+row,self.patch_size)
                col_filter = range(col,reduced_image.shape[1]+col,self.patch_size)

                # write values into target array
                target_array[:, row, col, :] = reduced_image[row_filter][:,col_filter,:].reshape(-1, reduced_image.shape[-1])

        return target_array

    def divide_into_test_training():
        """
        Divide the data into test and training split.
        """
        
        generator1 = torch.Generator().manual_seed(42)
        random_split(range(10), [3, 7], generator=generator1)