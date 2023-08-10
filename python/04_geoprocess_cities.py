import os
import shutil
import numpy as np
import subprocess
import glob
from urllib import urlretrieve
import zipfile
import pandas as pd
import json
import pickle

import geopandas as gpd
import fiona
import rasterio
from rasterio import features
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from osgeo import gdal, gdalnumeric, ogr, osr
from gdalconst import *
from PIL import Image, ImageDraw

from library.geoprocess import *
from library.cdb_imports import *

# load shapefile of all admin areas / countries as geodataframe
gdf = gpd.read_file('data/geo/countries/countries_nf2.shp'); gdf.head(3)

# filter out countries not internationally recognized
country_filter1 = gdf['WB_A3'] != '-99'
gdf = gdf.drop_duplicates(subset='WB_A3')
gdf = gdf[country_filter1].set_index('WB_A3')

# loop through rows of geodataframe and save each row as a country-specific shapefile in newly created dir
# shp_to_shps('data/geo/countries/shp', gdf)

# clip master raster from 2013 by each country shapefile to create country-level rasters
input_tif_path = 'data/geo/images/F182013.v4c_web.stable_lights.avg_vis.tif'
input_shp_dir = 'data/geo/countries/shp'
output_tif_dir = 'data/geo/countries/tif'
countries = [x.encode('UTF-8') for x in gdf.index.values]
# raster_to_rasters(countries, input_tif_path, input_shp_dir, output_tif_dir)

# polygonize rasters and save to target directory
input_tif_dir = 'data/geo/countries/tif'
output_shp_dir = 'data/geo/countries/poly'
# polygonize(input_tif_dir, output_shp_dir, countries)

# filter and union countries, save to target directory
input_dir = 'data/geo/countries/poly'
output_dir = 'data/geo/cities/union'
# union_and_filter(input_dir, output_dir, countries)

# split multi-polygons into polygons
input_dir = 'data/geo/cities/union'
output_dir = 'data/geo/cities/split'
# split_multi_to_single_poly(input_dir, output_dir)

# Merge shapefiles in directory
input_dir = 'data/geo/cities/split'
output_dir = 'data/geo/cities/merge'
output_filename = 'merged.shp'
# merge_shapefiles(input_dir, output_dir, output_filename)

# set CRS of merged shapefile
input_path = 'data/geo/cities/merge/merged.shp'
crs = 'epsg:4326'
output_path = 'data/geo/cities/merge/merged_crs.shp'
# set_crs(input_path, crs, output_path)

# zip merged shapefiles
target_dir = 'data/geo/cities/merge'
shp_filename = 'merged_crs.shp'
zip_filename = 'merged_crs.zip'

shp_path = os.path.join(target_dir, shp_filename)
zip_path = os.path.join(target_dir, zip_filename)
zip_path = os.path.abspath(zip_path)
shp_filename_no_ext = shp_filename[:-4]
glob_string = os.path.join(target_dir, shp_filename_no_ext) + '*'
list_of_shps = glob.glob(glob_string)
list_of_shps = [os.path.abspath(x) for x in list_of_shps]

#zip_files(list_of_shps, zip_path)

cdb_api_key = 'your_api_key'
cdb_domain = 'your_username'
c = CartoDBAPIKey(cdb_api_key, cdb_domain)
url = 'http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places_simple.zip'

# upload populated places shapefiles to cartodb
fi_1 = URLImport(url, c, privacy='public')

# upload zipped merged_crs shapefiles to cartodb
file_to_import = 'data/geo/cities/merge/merged_crs.zip'
fi_2 = FileImport(file_to_import, c, privacy='public')

#fi_1.run()
#fi_2.run()
#fi_1.success, fi_2.success

# call cartodb sql api to get polygons that intersect with cites, format as geojson
#intersection = c.sql(sql = 'with pop as (select nameascii, adm0_a3, pop_max, the_geom from ne_10m_populated_places_simple where pop_max > 1000000) select merged_crs.cartodb_id, pop.nameascii, pop.adm0_a3, pop.pop_max, merged_crs.the_geom from pop, merged_crs where ST_Within(pop.the_geom, merged_crs.the_geom)', format='geojson')

# dump geojson output from cartodb into file
dir_intersect = 'data/geo/cities/intersect'
filename = 'cities.geojson'
path = os.path.join(dir_intersect, filename)

#rm_and_mkdir(dir_intersect)
#with open(path, 'w') as outfile:
    #json.dump(intersection, outfile)

# write geojson to shapefile in same direcory: these are the metro clusters
shp_path = 'data/geo/cities/intersect/cities.shp'
geojson_path = 'data/geo/cities/intersect/cities.geojson'
#subprocess.check_call(['ogr2ogr', '-F', 'ESRI Shapefile', shp_path, geojson_path, 'OGRGeoJSON'])

# load cities shapefile and get zonal stats
tif_dir = 'data/geo/images'
input_shp_path = 'data/geo/cities/intersect/cities.shp'
#gdf = zonal_to_shp(tif_dir, shp_path)

# dump to pickle
#with open('data/geo/pickles/zonal_stats_m.pickle', 'wb') as f:
    #pickle.dump(gdf, f)

