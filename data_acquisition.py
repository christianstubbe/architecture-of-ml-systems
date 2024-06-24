import os
import time
import json
import pickle
import openeo
import numpy as np
import pyrosm as pyr
import geopandas as gpd
import matplotlib.pyplot as plt
from openeo.rest import OpenEoApiError
from openeo.processes import ProcessBuilder, if_, is_nan

import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask



class DataHandler: 
    def __init__(self, logger):
        """
        Initialize the DataHandler class and define openeo params.
        """
        self.logger = logger
        self.openeo_temporal_extent = ["2023-05-01", "2023-09-30"]
        self.openeo_bands = ["B04", "B03", "B02", "B08", "B12", "B11", "SCL"]
        self.openeo_max_cloud_cover = 30
        self.openeo_spatial_resolution = 10
        self.openeo_connection = None
        self.openeo_collections = None
        self.openeo_jobs = None
        
        if not os.path.exists("data"):
            os.makedirs("data")
            logger.info("Created data directory")
        else:
            logger.info("Data directory already exists")


    def create_directory(self, city: str):
        """
        Create a directory for each city.
        """
        os.makedirs(f"data/{city}", exist_ok=True)
        self.logger.info(f"{city}: Directory available")


    def get_buildings(self, city: str):
        """
        Return buildings for a given city
        """
        self.create_directory(city)
        
        # Check if local data for city is available
        if "buildings.geojson" in os.listdir(f"data/{city}"):
            self.logger.info(f"{city}: Using local building data")
            return gpd.read_file(f"data/{city}/buildings.geojson")

        # Download data for city
        fp = pyr.get_data(city, directory=os.path.join("data", city))
        osm = pyr.OSM(fp)
        self.logger.info(f"{city}: Downloaded data to data/{city}")

        # Get bounding box for city
        boundingbox = self.get_boundingbox(city, osm)

        # Get the buildings of the city
        buildings_geodf = osm.get_buildings()

        # Remove buildings outside of the bounding box of the city
        buildings_geodf = buildings_geodf.cx[boundingbox[0] : boundingbox[2], boundingbox[1] : boundingbox[3]]

        # Save the data of the city
        buildings_path = f"data/{city}/buildings.geojson"
        buildings_geodf.to_file(buildings_path, driver="GeoJSON")
        self.logger.info(f"{city}: Stored data to data/{city}/buildings.geojson")

        return buildings_geodf


    def get_boundingbox(self, city: str, osm = None):
        """
        Get the bounding box for a city.
        """

        # Return bounding box for Berlin as specified in exercise sheet to ensure correct testing results
        if city == "Berlin":
            return [13.294333, 52.454927, 13.500205, 52.574409]

        # Check if local bounds are available
        bounds_path = f"data/{city}/bounds.pkl"
        if os.path.exists(bounds_path):
            with open(bounds_path, "rb") as f:
                boundingbox = pickle.load(f)
            return boundingbox
        
        # Ensure OSM data is available 
        if osm is None:
            fp = pyr.get_data(city, directory=os.path.join("data", city))
            osm = pyr.OSM(fp)
            self.logger.info(f"{city}: Downloaded OSM data")

        # Get the boundaries
        geoframe_bounds = osm.get_boundaries()
        boundingbox = geoframe_bounds[geoframe_bounds["name"] == city].total_bounds

        # Check if bounding box is None
        if np.isnan(boundingbox[0]) or np.isnan(boundingbox[1]) or np.isnan(boundingbox[2]) or np.isnan(boundingbox[3]):
            self.logger.info(f"{city}: Bounding box is None. Using total bounds instead")
            boundingbox = geoframe_bounds.total_bounds
        self.logger.info(f"{city}: Bounding box is {boundingbox}")     

        # Save total bounds to pickle file
        with open(bounds_path, "wb") as f:
            pickle.dump(boundingbox, f)
        self.logger.info(f"{city}: Saved bounds to data/{city}/bounds.pkl")

        return boundingbox
    

    def get_satellite_image(self, city: str): 
        """
        Get satellite images for a city. Use local data if available.
        """
        if os.path.exists(f"data/{city}/openEO.tif"):
            self.logger.info(f"{city}: Using local satellite image")
            return rasterio.open(f"data/{city}/openEO.tif")
        else:
            self.download_satellite_image(city)
            return self.get_satellite_image(city)
    

    def connect_to_openeo(self):
        """
        Connect to the openEO backend and 
        """
        if self.openeo_connection is None:
            connection = openeo.connect("openeo.dataspace.copernicus.eu")
            connection.authenticate_oidc()
            self.openeo_connection = connection

            self.logger.info("Connected to openEO")
        else:
            self.logger.info("Already connected to openEO")


    def download_satellite_image(self, city: str):
        """
        Download satellite images for a city. Retry for 3 times if the job fails or takes longer than 30 min per job.
        """
        self.connect_to_openeo()
        
        # Log the currently running jobs
        self.logger.info("Current jobs:")
        for idx, job in enumerate(self.openeo_connection.list_jobs()):
            self.logger.info(f"{idx} {job['id']} {job['status']}")

        # Retry job up to 3 times. Raise exception after 3 retries.
        job_finished = False
        job_number_of_retries = 0
        while not job_finished : 
            if job_number_of_retries > 3:
                self.logger.error(f"{city}: Job failed after 3 retries")
                raise Exception(f"{city}: Job failed after 3 retries")
            job = self.create_and_start_openeo_job(city)    
            job_finished = self.await_job(city, job)
            job_number_of_retries += 1

        # Get job results and store in data/city
        job_results = self.openeo_connection.job(job.job_id).get_results()
        job_results.download_files(f"data/{city}")
        self.logger.info(f"{city}: Downloaded job results to data/{city}")


    def delete_jobs(self):
        """
        Delete all jobs on the openEO backend. Use only for debugging. 
        """
        self.connect_to_openeo()

        for idx, job in enumerate(self.openeo_connection.list_jobs()):
            self.logger.info(f"Deleteing job {idx}, {job["id"]}, {job["status"]}")
            self.openeo_connection.job(job["id"]).delete_job()


    def create_and_start_openeo_job(self, city: str, collection_id: str = "SENTINEL2_L2A"):
        """
        Creates an openeo processing job for a city and starts it.
        """
        # Transform order in boundingbox to dict
        boundingbox = self.get_boundingbox(city)
        boundingbox = {"west": boundingbox[0], "south": boundingbox[1], "east": boundingbox[2], "north": boundingbox[3]}
        
        # Create datacube
        datacube = self.openeo_connection.load_collection(
            collection_id=collection_id,
            spatial_extent=boundingbox,
            temporal_extent=self.openeo_temporal_extent,
            bands=self.openeo_bands,
            max_cloud_cover=self.openeo_max_cloud_cover,
        ).resample_spatial(self.openeo_spatial_resolution)

        # Create cloud mask
        scl = datacube.band("SCL")

        # Filter out cloud median probability, cloud high probability, and snow/ice
        mask = (scl == 8) | (scl == 9) | (scl == 11)

        # Resample mask to the spatial resolution of the datacube
        mask = mask.resample_cube_spatial(datacube.band("B04"))
        
        # Create the RGB image
        datacube_rgbFU = datacube.filter_bands(self.openeo_bands[:-1])
        
        # Apply cloud mask
        datacube_rgb_masked = datacube_rgbFU.mask(mask)
        
        # Reduce temporal to median 
        datacube_rgb_masked_reduced_t = datacube_rgb_masked.reduce_temporal("median")

        # Define image format 
        datacube_for_submission = datacube_rgb_masked_reduced_t.save_result(format="GTiff")
        
        # Create openEO job with datacube
        job = datacube_for_submission.create_job(title=f"{city}__pic")
        self.logger.info(f"{city}: Created openEO job")

        # Start openEO job
        job.start_job()
        self.logger.info(f"{city}: Started openEO job with ID: {job.job_id}")        

        return job


    def await_job(self, city, job):
        """
        Awaits the processing of a openeo job. 
        Returns when the job is finished or raises an exception if the job failed.
        """

        for i in range(30):
            status = self.openeo_connection.job(job.job_id).status()
            self.logger.debug(f"{city}: Job {job.job_id} status: {status}")
          
            if status == "finished":
                self.logger.info(f"{city}: Job {job.job_id} finished")
                return True
            
            elif status == "error":
                self.logger.warning(f"{city}: Job {job.job_id} failed. Trying again.")
                return False            
            
            time.sleep(60)
        self.logger.error(f"{city}: Job {job.job_id} did not finish in time")
        return False

    def get_building_mask(self, city: str):  
        """
        Get the local building mask for buildings in a city.
        """
        # Check if the building mask is already available
        if os.path.exists(f"data/{city}/building_mask.tif"):
            self.logger.info(f"{city}: Using local building mask")
            return rasterio.open(f"data/{city}/building_mask.tif").read(1)

        # Create new building mask 
        satellite_image = self.get_satellite_image(city)

        # Get satellite image metadata
        transform = satellite_image.transform
        out_shape = (satellite_image.height, satellite_image.width)
        crs = satellite_image.crs

        # Read the GeoJSON file with building polygons
        buildings = self.get_buildings(city)
        buildings = buildings.to_crs(crs)  # Ensure the CRS matches the GeoTIFF

        # Create a mask where pixels inside buildings are True, others are False
        mask = geometry_mask(
            buildings.geometry, transform=transform, invert=True, out_shape=out_shape
        )

        # Store the mask as a GeoTIFF file
        
        out_meta = satellite_image.meta
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mask.shape[0],
                "width": mask.shape[1],
                # "transform": transform,
                "count": 1,
            }
        )

        # boolmask is automatically being saved as int16 [0,1]
        with rasterio.open(f"data/{city}/building_mask.tif", "w", **out_meta) as dest:
            dest.write(mask, indexes=1)

        return mask


    # TODO: Finish with Joscha.
    def plot(self, city: str, backend: str = "matplotlib"):
        """
        Plot the data for a city either with matplotlib or plotly.
        """
        
        if backend != "matplotlib":
            raise NotImplementedError("Only matplotlib is supported at the moment")
        
        buildings = self.get_buildings(city)
        satellite_image = self.get_satellite_image(city)
        mask = self.get_building_mask(city)
        
        # Design plots
        fig, ax = plt.subplots(figsize=(10, 10))
        buildings.plot(ax=ax, color="black")
        plt.title(f"{city} buildings")
        plt.axis("off")

        # Create RGB image
        red = satellite_image.read(1)
        green = satellite_image.read(2)
        blue = satellite_image.read(3)
        
        # apply histogram stretching
        def stretch_hist(band):
            p2, p98 = np.percentile(band, (0.5, 99.5))
            return np.clip((band - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)


        red_stretched = stretch_hist(red)
        green_stretched = stretch_hist(green)
        blue_stretched = stretch_hist(blue)

        print(red_stretched.shape, green_stretched.shape, blue_stretched.shape)
        # Stack the bands after stretching
        rgb_stretched = np.dstack((red_stretched, green_stretched, blue_stretched))

        # Plot the histogram-stretched RGB image
        plt.figure(figsize=figure_size)
        plt.imshow(rgb_stretched)
        # plt.title("Histogram Stretched RGB Composite Image")
        plt.title("RGB Bands from Sentinel-2 L2A")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_RGB.png"))
        plt.close()


        # RGB image with higher brightness
        red_norm = (red - np.min(red)) / (np.max(red) - np.min(red))
        green_norm = (green - np.min(green)) / (np.max(green) - np.min(green))
        blue_norm = (blue - np.min(blue)) / (np.max(blue) - np.min(blue))
        pseudo_RGB_image = np.dstack((red_norm, green_norm, blue_norm))

        pseudo_RGB_image_normalized = (pseudo_RGB_image - np.min(pseudo_RGB_image)) / (
            pseudo_RGB_image.max() - pseudo_RGB_image.min()
        )


        pseudo_RGB_image_brighter = pseudo_RGB_image_normalized * brightness
        pseudo_RGB_image_brighter = np.clip(pseudo_RGB_image_brighter, 0, 1)
        plt.figure(figsize=figure_size)
        plt.imshow(pseudo_RGB_image_brighter)
        plt.title("RGB Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_RGB_Brighter.png"))
        plt.close()

        # single band img
        # single_band = dataset.read(1)
        single_band_stretched = stretch_hist(dataset.read(1))
        plt.figure(figsize=figure_size)
        plt.imshow(single_band_stretched, cmap="gray")
        plt.title("Single Band Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_SingleBand.png"))
        plt.close()


        # B8 B4 B3 -> False Color
        b8 = dataset.read(4)
        b8_stretched = stretch_hist(b8)
        b4 = red_stretched
        b3 = green_stretched

        false_color = np.dstack((b8_stretched, b4, b3))
        plt.figure(figsize=figure_size)
        plt.imshow(false_color)
        plt.title("False Color Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_FalseColor.png"))
        plt.close()

        # params["bands"] = ["B04", "B03", "B02", "B08", "B12", "B11", "SCL"] # scl must be last

        # B12, B11, B4 -> False Color Urban
        b12 = dataset.read(5)
        b11 = dataset.read(6)
        b04 = dataset.read(1)
        b12_norm = (b12 - np.min(b12)) / (np.max(b12) - np.min(b12))
        b11_norm = (b11 - np.min(b11)) / (np.max(b11) - np.min(b11))
        b04_norm = (b04 - np.min(b04)) / (np.max(b04) - np.min(b04))


        false_color_urban = np.dstack((b12_norm, b11_norm, b04_norm)) * brightness
        false_color_urban = np.clip(false_color_urban, 0, 1)

        plt.figure(figsize=figure_size)
        plt.imshow(false_color_urban)
        plt.title("False Color Urban Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_FalseColorUrban.png"))
        plt.close()


        # get vegetation_index
        def vegetation_index(band1, band2):
            return (band1 - band2) / (band1 + band2)


        ndvi = vegetation_index(dataset.read(4), dataset.read(3))
        plt.figure(figsize=figure_size)
        plt.imshow(ndvi, cmap="RdYlGn")
        plt.title("NDVI Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_NDVI.png"))
        plt.close()

        # Visualize the mask
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap="Blues")
        plt.title("Building Mask")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_BuildingMask.png"))
        plt.close()


        # Load the image
        img = single_band_stretched  # Assuming `blue_stretched` is the single band image
        with rasterio.open("data/BerlinTest/building_mask.tif") as ds_mask:
            mask = ds_mask.read(1)

        blue_cmap = plt.cm.Blues
        blue_building_mask = blue_cmap(mask / mask.max())
        blue_building_mask[..., 3] = mask * 0.8

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap="gray", alpha=1)

        plt.imshow(blue_building_mask)

        # Set the title and axis labels
        plt.title("Image with Buildings Mask")
        plt.axis("off")

        # Show the plot
        # plt.show()
        plt.savefig(os.path.join(image_path_berlin, "BerlinTest_BuildingMaskOverlay.png"))
        plt.close()