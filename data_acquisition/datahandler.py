# basics
import os
import time
import json
import pickle
import openeo
import numpy as np

# geography
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask


# download
import pyrosm as pyr
from openeo.rest import OpenEoApiError
from openeo.processes import ProcessBuilder, if_, is_nan


# plotting
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utilities.utils import stretch_hist


class DataHandler:
    def __init__(self, logger, path_to_data_directory="data"):
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
        self.openeo_jobs = None  #
        self.path_to_data_directory = path_to_data_directory

        if not os.path.exists(self.path_to_data_directory):
            os.makedirs(self.path_to_data_directory)
            logger.info("Created data directory")
        else:
            logger.info("Data directory already exists")

    def create_directory(self, city: str):
        """
        Create a directory for each city.
        """
        os.makedirs(os.path.join(self.path_to_data_directory, city), exist_ok=True)
        self.logger.info(f"{city}: Directory available")

    def get_OSM_data(self, city: str):
        # Download data for city
        fp = pyr.get_data(
            city, directory=os.path.join(self.path_to_data_directory, city)
        )
        osm = pyr.OSM(fp)
        self.logger.info(
            f"{city}: Downloaded data to {self.path_to_data_directory}/{city}"
        )
        return osm

    def get_buildings(self, city: str):
        """
        Return buildings for a given city
        """
        self.create_directory(city)

        # Check if local data for city is available
        if "buildings.geojson" in os.listdir(
            os.path.join(self.path_to_data_directory, city)
        ):
            self.logger.info(f"{city}: Using local building data")
            return gpd.read_file(
                os.path.join(self.path_to_data_directory, city, "buildings.geojson")
            )

        # get osm object
        osm = self.get_OSM_data(city)

        # Get bounding box for city
        boundingbox = self.get_boundingbox(city, osm)

        # Get the buildings of the city
        buildings_geodf = osm.get_buildings()

        # Remove buildings outside of the bounding box of the city
        buildings_geodf = buildings_geodf.cx[
            boundingbox[0] : boundingbox[2], boundingbox[1] : boundingbox[3]
        ]

        # Save the data of the city
        buildings_path = os.path.join(
            self.path_to_data_directory, city, "buildings.geojson"
        )
        buildings_geodf.to_file(buildings_path, driver="GeoJSON")
        self.logger.info(f"{city}: Stored data to {buildings_path}")

        return buildings_geodf

    def get_boundingbox(self, city: str, osm=None):
        """
        Get the bounding box for a city.
        """

        # Return bounding box for Berlin as specified in exercise sheet to ensure correct testing results
        if city == "Berlin":
            return [13.294333, 52.454927, 13.500205, 52.574409]

        # Check if local bounds are available
        bounds_path = os.path.join(self.path_to_data_directory, city, "bounds.pkl")
        if os.path.exists(bounds_path):
            with open(bounds_path, "rb") as f:
                boundingbox = pickle.load(f)
            return boundingbox

        # Ensure OSM data is available
        if osm is None:
            self.get_buildings(city=city)

        # Get the boundaries
        geoframe_bounds = osm.get_boundaries()
        boundingbox = geoframe_bounds[geoframe_bounds["name"] == city].total_bounds

        # Check if bounding box is None
        if (
            np.isnan(boundingbox[0])
            or np.isnan(boundingbox[1])
            or np.isnan(boundingbox[2])
            or np.isnan(boundingbox[3])
        ):
            self.logger.info(
                f"{city}: Bounding box is None. Using total bounds instead"
            )
            boundingbox = geoframe_bounds.total_bounds
        self.logger.info(f"{city}: Bounding box is {boundingbox}")

        # Save total bounds to pickle file
        with open(bounds_path, "wb") as f:
            pickle.dump(boundingbox, f)
        self.logger.info(f"{city}: Saved bounds to {bounds_path}")

        # save boundaries to geojson file
        geoframe_bounds.to_file(
            os.path.join(self.path_to_data_directory, city, "boundaries.geojson"),
            driver="GeoJSON",
        )
        self.logger.info(
            f"{city}: Saved boundaries to {os.path.join(self.path_to_data_directory, city,'boundaries.geojson')}"
        )

        return boundingbox

    def get_satellite_image(self, city: str, return_rasterio_dataset=False):
        """
        Get satellite images for a city. Use local data if available. Returns an Array with (H, W, C) shape
        """
        if os.path.exists(
            os.path.join(self.path_to_data_directory, city, "openEO.tif")
        ):
            self.logger.info(f"{city}: Using local satellite image")
            ds = rasterio.open(
                os.path.join(self.path_to_data_directory, city, "openEO.tif")
            )
            if return_rasterio_dataset:
                return ds

            # Read all channels
            sat_data = ds.read()

            # Transpose to (H, W, C)
            sat_data = np.transpose(sat_data, (1, 2, 0))
            return sat_data
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
        while not job_finished:
            if job_number_of_retries > 3:
                self.logger.error(f"{city}: Job failed after 3 retries")
                raise Exception(f"{city}: Job failed after 3 retries")
            job = self.create_and_start_openeo_job(city)
            job_finished = self.await_job(city, job)
            job_number_of_retries += 1

        # Get job results and store in data/city
        job_results = self.openeo_connection.job(job.job_id).get_results()
        job_results.download_files(os.path.join(self.path_to_data_directory, city))
        self.logger.info(
            f"{city}: Downloaded job results to {os.path.join(self.path_to_data_directory, city)}"
        )

    def delete_jobs(self):
        """
        Delete all jobs on the openEO backend. Use only for debugging.
        """
        self.connect_to_openeo()

        for idx, job in enumerate(self.openeo_connection.list_jobs()):
            self.logger.info(f"Deleting job {idx}, {job['id']}, {job['status']}")
            self.openeo_connection.job(job["id"]).delete_job()

    def get_jobs(self):
        """
        Get all jobs on the openEO backend.
        """
        self.connect_to_openeo()
        self.openeo_jobs = self.openeo_connection.list_jobs()
        self.logger.info("Current jobs:")
        for idx, job in enumerate(self.openeo_jobs):
            self.logger.info(f"{idx} {job['id']} {job['status']}")
        return self.openeo_jobs

    def create_and_start_openeo_job(
        self, city: str, collection_id: str = "SENTINEL2_L2A"
    ):
        """
        Creates an openeo processing job for a city and starts it.
        """
        # Transform order in boundingbox to dict
        boundingbox = self.get_boundingbox(city)
        boundingbox = {
            "west": boundingbox[0],
            "south": boundingbox[1],
            "east": boundingbox[2],
            "north": boundingbox[3],
        }

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
        datacube_for_submission = datacube_rgb_masked_reduced_t.save_result(
            format="GTiff"
        )

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

        for i in range(60):
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

    def get_building_mask(
        self, city: str, loaded_buildings=None, all_touched: bool = False
    ):
        """
        Get the local building mask for buildings in a city.
        """
        if all_touched:
            filename = "building_mask_dense"
        else:
            filename = "building_mask_sparse"
        # Check if the building mask is already available
        if os.path.exists(
            os.path.join(self.path_to_data_directory, city, f"{filename}.tif")
        ):
            self.logger.info(f"{city}: Using local building mask")
            mask = rasterio.open(
                os.path.join(self.path_to_data_directory, city, f"{filename}.tif")
            ).read(1)
            if mask.sum() == 0:
                os.remove(
                    os.path.join(self.path_to_data_directory, city, f"{filename}.tif")
                )
                return self.get_building_mask(city, loaded_buildings, all_touched)
            return mask

        # Create new building mask
        satellite_image = self.get_satellite_image(city, return_rasterio_dataset=True)

        # Get satellite image metadata
        transform = satellite_image.transform
        out_shape = (satellite_image.height, satellite_image.width)
        crs = satellite_image.crs

        # Read the GeoJSON file with building polygons
        if loaded_buildings is not None:
            buildings = loaded_buildings
        else:
            buildings = self.get_buildings(city)
            buildings = buildings.to_crs(crs)  # Ensure the CRS matches the GeoTIFF

        # Create a mask where pixels inside buildings are True, others are False
        # TODO all_touched paramer nutzen für zweite Maske
        mask = geometry_mask(
            buildings.geometry,
            transform=transform,
            invert=True,
            out_shape=out_shape,
            all_touched=all_touched,
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

        with rasterio.open(
            os.path.join(self.path_to_data_directory, city, f"{filename}.tif"),
            "w",
            **out_meta,
        ) as dest:
            dest.write(mask, indexes=1)

        # create the boundaries mask for the city
        # shows what part of the image is covered with labels
        # from utilities.data_processing import create_boundaries_mask
        # check if boundaries mask is already available
        if os.path.exists(
            os.path.join(self.path_to_data_directory, city, "boundaries_mask.tif")
        ):
            self.logger.info(f"{city}: Using local boundaries mask")
        else:
            self.logger.info(f"{city}: Creating boundaries mask")
            create_boundaries_mask(
                os.path.join(self.path_to_data_directory, city),
                crs,
                satellite_image.meta,
                transform,
                out_shape,
                out_file_name="boundaries_mask.tif",
            )

        return mask

    # def boundings_geojson(path: str):
    # # open the osm.pbf file inside the city directory
    #     osm_pbf = [f for f in os.listdir(path) if f.endswith(".osm.pbf")][0]
    #     print(osm_pbf)
    #     osm = pyr.OSM(os.path.join(path, osm_pbf))

    #     geoframe_bounds = osm.get_boundaries()

    #     print(os.path.join(path, "boundaries.geojson"))
    #     geoframe_bounds.to_file(
    #         os.path.join(path, "boundaries.geojson"),
    #         driver="GeoJSON"
    # )

    def get_boundaries_mask(self, city, out_file_name="boundaries_mask.tif"):
        """
        create a mask of where the boundaries of the city are,
        so patches where no building were downloaded but that are included in the satllite image can be excluded
        """
        if os.path.exists(
            os.path.join(self.path_to_data_directory, city, out_file_name)
        ):
            self.logger.info(f"{city}: Using local boundary mask")
            mask = rasterio.open(
                os.path.join(self.path_to_data_directory, city, out_file_name)
            ).read(1)
            return mask

        satellite_image = self.get_satellite_image(city, return_rasterio_dataset=True)
        crs = satellite_image.crs
        meta = satellite_image.meta
        transform = satellite_image.transform
        out_shape = (satellite_image.height, satellite_image.width)

        # get osm object
        osm = self.get_OSM_data(city)
        boundaries = osm.get_boundaries()

        bounds_poly = boundaries.to_crs(crs)

        mask = rasterio.features.geometry_mask(
            bounds_poly.geometry,
            transform=transform,
            invert=True,
            out_shape=out_shape,
            all_touched=True,
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
        with rasterio.open(
            os.path.join(self.path_to_data_directory, city, out_file_name), "w", **meta
        ) as dest:
            dest.write(mask, indexes=1)
        return mask

    def plot(
        self,
        city: str = "BerlinTest",
        backend: str = "matplotlib",
        figure_size: tuple = (10, 10),
        brightness: int = 5,
        image_directory: str = "img/",
        show_plot: bool = False,
        slice_to_be_plotted=None,
        mask="sparse",
    ):
        """
        Plot the data for a city either with matplotlib or plotly.
        """

        if backend != "plotly" and backend != "matplotlib":
            raise NotImplementedError(
                "Only matplotlib and plotly is supported at the moment"
            )

        satellite_data = self.get_satellite_image(city)
        mask = self.get_building_mask(city, all_touched=mask == "dense")
        # Take out slice if only a slice is to be plotted
        if slice_to_be_plotted is not None:
            satellite_data = satellite_data[
                slice_to_be_plotted[0], slice_to_be_plotted[1]
            ]
            mask = mask[slice_to_be_plotted[0], slice_to_be_plotted[1]]

        if backend == "matplotlib":
            # load buildings
            buildings = self.get_buildings(city)

            # create image out path
            image_path_out = os.path.join(image_directory, city)
            # make the output directory if not exists
            os.makedirs(image_path_out, exist_ok=True)

            # Design plots
            fig, ax = plt.subplots(figsize=figure_size)
            buildings.plot(ax=ax, color="black")
            plt.title(f"{city} buildings")
            plt.axis("off")

        # RGB Bands from Sentinel 2
        red = satellite_data[..., 0]
        green = satellite_data[..., 1]
        blue = satellite_data[..., 2]

        # Apply histogram stretching
        red_stretched = stretch_hist(red)
        green_stretched = stretch_hist(green)
        blue_stretched = stretch_hist(blue)

        # Stack the bands after stretching
        rgb_stretched = np.dstack((red_stretched, green_stretched, blue_stretched))

        if backend == "matplotlib":
            # Plot the histogram-stretched RGB image
            plt.figure(figsize=figure_size)
            plt.imshow(rgb_stretched)
            # plt.title("Histogram Stretched RGB Composite Image")
            plt.title(f"{city} RGB Bands from Sentinel-2 L2A")
            plt.axis("off")
            # plt.show()
            plt.savefig(os.path.join(image_path_out, f"{city}_RGB.png"))
            if show_plot:
                plt.show()
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

        if backend == "matplotlib":
            plt.figure(figsize=figure_size)
            plt.imshow(pseudo_RGB_image_brighter)
            plt.title(f"{city} RGB Image")
            plt.axis("off")
            # plt.show()
            plt.savefig(os.path.join(image_path_out, f"{city}_RGB_Brighter.png"))
            if show_plot:
                plt.show()
            plt.close()

            # single band img
            # single_band = satellite_image.read(1)
            single_band_stretched = stretch_hist(red)
            plt.figure(figsize=figure_size)
            plt.imshow(single_band_stretched, cmap="gray")
            plt.title(f"{city} Single Band Image")
            plt.axis("off")
            # plt.show()
            plt.savefig(os.path.join(image_path_out, f"{city}_SingleBand.png"))
            if show_plot:
                plt.show()
            plt.close()
        elif backend == "plotly":

            # plot the mask
            fig = px.imshow(mask.astype(np.uint8), binary_string=True)

            # Overlay the mask with the image
            fig.add_trace(
                go.Image(
                    z=(pseudo_RGB_image_brighter * 255).astype(np.uint8), opacity=1
                )
            )

            # Update layout with a button to toggle mask visibility
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list(
                            [
                                dict(
                                    args=[{"opacity": [0, 1]}],
                                    label="Hide Mask",
                                    method="restyle",
                                ),
                                dict(
                                    args=[{"opacity": [0.5, 0.5]}],
                                    label="Show Mask",
                                    method="restyle",
                                ),
                            ]
                        ),
                    ),
                ],
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )

            # Enable zooming and panning
            fig.update_xaxes(constrain="domain")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_layout(height=1000, width=1000)

            # Display the figure
            return fig

        # B8 B4 B3 -> False Color
        b8 = satellite_data[..., 3]
        b8_stretched = stretch_hist(b8)
        b4 = red_stretched
        b3 = green_stretched

        false_color = np.dstack((b8_stretched, b4, b3))
        plt.figure(figsize=figure_size)
        plt.imshow(false_color)
        plt.title(f"{city} False Color Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_out, f"{city}_FalseColor.png"))
        if show_plot:
            plt.show()
        plt.close()

        # params["bands"] = ["B04", "B03", "B02", "B08", "B12", "B11", "SCL"] # scl must be last

        # B12, B11, B4 -> False Color Urban
        b12 = satellite_data[..., 4]
        b11 = satellite_data[..., 5]
        b04 = red
        b12_norm = (b12 - np.min(b12)) / (np.max(b12) - np.min(b12))
        b11_norm = (b11 - np.min(b11)) / (np.max(b11) - np.min(b11))
        b04_norm = (b04 - np.min(b04)) / (np.max(b04) - np.min(b04))

        false_color_urban = np.dstack((b12_norm, b11_norm, b04_norm)) * brightness
        false_color_urban = np.clip(false_color_urban, 0, 1)

        plt.figure(figsize=figure_size)
        plt.imshow(false_color_urban)
        plt.title(f"{city} False Color Urban Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_out, f"{city}_FalseColorUrban.png"))
        if show_plot:
            plt.show()
        plt.close()

        # get vegetation_index
        def vegetation_index(band1, band2):
            return (band1 - band2) / (band1 + band2)

        ndvi = vegetation_index(satellite_data[..., 3], satellite_data[..., 2])
        plt.figure(figsize=figure_size)
        plt.imshow(ndvi, cmap="RdYlGn")
        plt.title(f"{city} NDVI Image")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_out, f"{city}_NDVI.png"))
        if show_plot:
            plt.show()
        plt.close()

        # Visualize the mask
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap="Blues")
        plt.title(f"{city} Building Mask")
        plt.axis("off")
        # plt.show()
        plt.savefig(os.path.join(image_path_out, f"{city}_BuildingMask.png"))
        if show_plot:
            plt.show()
        plt.close()

        # Load the image
        img = (
            single_band_stretched  # Assuming `blue_stretched` is the single band image
        )
        blue_cmap = plt.cm.Blues
        blue_building_mask = blue_cmap(mask / mask.max())
        blue_building_mask[..., 2] = mask * 0.8

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap="gray", alpha=1)

        plt.imshow(blue_building_mask)

        # Set the title and axis labels
        plt.title(f"{city} Image with Buildings Mask")
        plt.axis("off")

        # Show the plot
        # plt.show()
        plt.savefig(os.path.join(image_path_out, f"{city}_BuildingMaskOverlay.png"))
        if show_plot:
            plt.show()
        plt.close()
