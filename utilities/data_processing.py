import os

import rasterio
import geopandas as gpd
import pyrosm as pyr


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

def create_boundaries_mask(path, crs, meta, transform, out_shape, out_file_name="boundaries_mask.tif"):
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
    with rasterio.open(os.path.join(path, out_file_name), "w", **meta) as dest:
            dest.write(mask, indexes=1)