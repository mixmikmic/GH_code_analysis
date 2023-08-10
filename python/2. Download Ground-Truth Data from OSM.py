import json

import shapely.geometry
import geopandas as gpd
import geopandas_osm.osm

meta_df = gpd.read_file('vectors/image_metadata.geojson')
poly = shapely.geometry.box(*meta_df.unary_union.bounds)

osm_df = geopandas_osm.osm.query_osm('way', poly, recurse='down', tags='building')
building_columns = osm_df.columns

buildings = osm_df[~osm_df.building.isnull()][['building', 'name', 'geometry']]
building_centroids = buildings.set_geometry(buildings.centroid, inplace=False)
building_centroids.to_file('vectors/building_centers.geojson', 'GeoJSON')

from collections import Counter

Counter(building_centroids.building.values)

osm_df = geopandas_osm.osm.query_osm('way', poly, recurse='down', tags='landuse')
landuse = osm_df[~osm_df.landuse.isnull()]
print(landuse.landuse.unique())
print(landuse.shape)
landuse.to_file('vectors/landuse.geojson', 'GeoJSON')

osm_df = geopandas_osm.osm.query_osm('way', poly, recurse='down', tags='waterway')
waterways = osm_df[~osm_df.waterway.isnull()]
print(waterways.waterway.unique())
print(waterways.shape)
waterways.to_file('vectors/waterways.geojson', 'GeoJSON')

osm_df = geopandas_osm.osm.query_osm('way', poly, recurse='down', tags='natural')
nature = osm_df[~osm_df.natural.isnull()]
print(nature.natural.unique())
print(nature.shape)
nature.to_file('vectors/nature.geojson', 'GeoJSON')

import os

import rasterio

image_dir_path = '/mnt/OSN_Data/spacenet/AOI_2_Vegas_Train/MUL-PanSharpen'

tiff_images = [os.path.join(image_dir_path, image_name) 
               for image_name in os.listdir(image_dir_path) 
               if image_name.lower().endswith('tif')]

res = []
for image_file_name in tiff_images:
    with rasterio.open(image_file_name) as src:
        bbox = shapely.geometry.box(*src.bounds)
    res.append((image_file_name, bbox))
image_summary = gpd.GeoDataFrame(res, columns=['image_name', 'geometry'], crs={'init': 'epsg:4326'})
poly = shapely.geometry.box(*image_summary.unary_union.bounds)

# tag = 'generator:source=solar'
osm_df = geopandas_osm.osm.query_osm('way', poly, recurse='down', tags='power')
osm_df.columns

osm_df['generator:type'].unique()

(osm_df['generator:type'] == 'solar_photovoltaic_panel').sum()

osm_df['generator:source'].unique()

(osm_df['generator:source'] == 'solar').sum()

osm_df['generator:method'].unique()

(osm_df['generator:method'] == 'photovoltaic').sum()

is_solar = (osm_df['generator:type'] == 'solar_photovoltaic_panel') | (osm_df['generator:source'] == 'solar') | (osm_df['generator:method'] == 'photovoltaic')
solar_panels = osm_df[is_solar]
solar_images = image_summary[image_summary.intersects(solar_panels.unary_union)]
len(solar_images)

import numpy as np

with rasterio.open(solar_images.iloc[0].image_name) as src:
        img = src.read([5, 3, 2]).transpose([1,2,0])
        img_bounds = shapely.geometry.box(*src.bounds)
        img_transform = list(np.array(~src.transform)[[0, 1, 3, 4, 2, 5]])


panel_geometry = solar_panels[solar_panels.intersects(img_bounds)].unary_union
panel_geometry_imcoords = shapely.affinity.affine_transform(panel_geometry, img_transform)

def scale_bands(img, lower_pct = 1, upper_pct = 99):
    """
    Rescale the bands of a multichannel image
    """
    # Loop through the image bands, rescaling each one
    img_scaled = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        lower, upper = np.percentile(band, [lower_pct, upper_pct])
        band = (band - lower) / (upper - lower) * 255
        img_scaled[:, :, i] = np.clip(band, 0, 255).astype(np.uint8)
    return img_scaled

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(scale_bands(img))


ax.plot(*panel_geometry_imcoords.xy, linewidth=2)

