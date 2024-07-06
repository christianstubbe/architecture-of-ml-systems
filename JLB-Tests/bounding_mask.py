# %%
cities = ["Aachen", 'London', 'CapeTown', 'Hamburg', 'Johannesburg', 'London', 'Montreal', 'Paris', 'Seoul', 'Singapore', 'Sydney']

parameters = {
    "abs_path": "/home/jlb/Projects/architecture-of-ml-systems/data",
}
# %%
# basics
import os
import utils
import numpy as np
from tqdm.notebook import tqdm 
import pyrosm as pyr
import rasterio

# torch
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import lightning as L




# custom modules
from data_acquisition import DataHandler
from data_preparation import apply_preprocessing_pipeline
import geopandas as gpd
logger = utils.setup_logger(level='ERROR')
datahandler = DataHandler(logger, path_to_data_directory="data")

# %%

def boundings_geojson(path: str):
    # open the osm.pbf file inside the city directory
    osm_pbf = [f for f in os.listdir(path) if f.endswith(".osm.pbf")][0]
    print(osm_pbf)
    osm = pyr.OSM(os.path.join(path, osm_pbf))

    geoframe_bounds = osm.get_boundaries()

    print(os.path.join(path, "boundaries.geojson"))
    geoframe_bounds.to_file(
        os.path.join(path, "boundaries.geojson"),
        driver="GeoJSON"
    )

def create_boundaries_mask(path, crs, meta, transform, out_shape):
    """
    Pass path that must include a osm.pbf file
    """
    # open the osm.pbf file inside the city directory
    osm_pbf = [f for f in os.listdir(path) if f.endswith(".osm.pbf")][0]
    boundaries = pyr.OSM(os.path.join(path, osm_pbf)).get_boundaries()

    bounds_poly = boundaries.to_crs(crs)

    mask = rasterio.features.geometry_mask(
        bounds_poly.geometry, 
        transform=transform, 
        invert=True, 
        out_shape=out_shape, 
        all_touched=True
    )
    meta.update(
        {
            "driver": "GTiff",
            "height": mask.shape[0],
            "width": mask.shape[1],
            # "transform": transform,
            "count": 1,
        }
    )
    with rasterio.open(os.path.join(path, "boundaries_mask.tif"), "w", **meta) as dest:
            dest.write(mask, indexes=1)

    
# %%
cities = [d for d in os.listdir(parameters["abs_path"]) if os.path.isdir(os.path.join(parameters["abs_path"], d))]
cities.sort()
for city in cities:
    print(city)
    path = os.path.join(parameters["abs_path"], city)
    print(path)
    # check if boundaries.geojson exists
    if "boundaries.geojson" in os.listdir(path):
        print("boundaries.geojson exists")
    else:
        boundings_geojson(path)
    
    # load the boundaries.geojson
    print(f"loading {os.path.join(path, 'boundaries.geojson')}")
    boundaries = gpd.read_file(os.path.join(path, "boundaries.geojson"))
    
    # create a boundaries map



    



# %%
import matplotlib.pyplot as plt
import geopandas as gpd
# Read the boundaries.geojson file
boundaries = gpd.read_file(os.path.join(path, "boundaries.geojson"))

# Plot the boundaries

boundaries.plot()


# Show the plot
plt.show()