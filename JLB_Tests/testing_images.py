# %%
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio


parameters = {
    "abs_data_path": "/home/jlb/Projects/architecture-of-ml-systems/data",
    "building_geojson": "buildings.geojson",
    "building_mask": "building_mask.tif",
}

dirs = [d for d in os.listdir(parameters["abs_data_path"]) if os.path.isdir(os.path.join(parameters["abs_data_path"], d))]
print(dirs)
# %%

def create_buildings_mask(city_name: str, geotiff_path: str, building_geojson: str):
    pass
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs
# %%
def plot_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap="Blues")
        plt.title(f"Building Mask {mask_path.split('/')[-2]}")
        # plt.axis("off")
        plt.show()
# %%
for city in dirs:
    mask_path = os.path.join(parameters["abs_data_path"], city, parameters["building_mask"])
    print(mask_path)
    # check if path exists:
    if os.path.exists(mask_path):
        plot_mask(mask_path)
    else:
        print(f"Mask for {city} does not exist.")    
