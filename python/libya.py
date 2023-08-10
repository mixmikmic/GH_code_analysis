import os
import json
import requests
from functools import partial
import pyproj
import geopyspark as gps

from pyspark import SparkContext
from geonotebook.wrappers import TMSRasterData, VectorData
from shapely.geometry import shape, MultiPoint, MultiLineString
from shapely.ops import transform

conf = gps.geopyspark_conf(appName="Libya Weighted Overlay", master="local[*]")
conf.set("spark.hadoop.yarn.timeline-service.enabled", False)
pysc = SparkContext.getOrCreate(conf)

libya_roads_json = requests.get('https://s3.amazonaws.com/geopyspark-demo/libya/roads.geojson').json()
libya_roads = MultiLineString([shape(geom['geometry']) for geom in libya_roads_json['features']])

# All execution time here is sending WKB over py4j socket
ro = gps.RasterizerOptions(includePartial=True, sampleType='PixelIsArea')

road_raster = gps.rasterize(geoms=libya_roads.geoms, 
                            crs="EPSG:3857",
                            zoom=8, 
                            fill_value=1,
                            cell_type=gps.CellType.FLOAT32,
                            options=ro,
                            num_partitions=20)

road_raster.layer_metadata.bounds

# Pyramid up from base layer
road_pp = road_raster.pyramid(resample_method=gps.ResampleMethod.MAX).cache()

# color map roads 1 to red
roads_cm = gps.ColorMap.from_colors(breaks=[1], color_list=[0xff000080])

# start JVM tile server and serve tiles to map
server = gps.TMS.build(source=road_pp, display=roads_cm)
M.add_layer(TMSRasterData(server), name="TMS")

M.remove_layer(M.layers[0])

# road network will shape our friction layer
road_friction = road_raster.reclassify(value_map={1:1},
                                       data_type=int,
                                       replace_nodata_with=10)

# starting points for cost distance operation

population_json = requests.get('https://s3.amazonaws.com/geopyspark-demo/libya/population.geojson').json()
population_centers = MultiPoint([shape(geom['geometry']) for geom in population_json['features']])

conflict_json = requests.get('https://s3.amazonaws.com/geopyspark-demo/libya/conflict.geojson').json()
conflict_centers = MultiPoint([shape(feature['geometry']) for feature in conflict_json['features'] if feature['geometry'] != None])

conflict_centers

# Convert population centers data from EPSG:3857 to EPSG:4326 for display on map
project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:3857'),
    pyproj.Proj(init='epsg:4326'))

population_4326 = transform(project, population_centers)

# Write reprojected data to file

if 'VIRTUAL_ENV' in os.environ:
    get_ipython().system('pip3 install geojson')
else:
    get_ipython().system('pip3 install --user geojson')
    
import geojson

with open('/tmp/population-4326.geojson', 'w') as f:
    geojson.dump(geojson.Feature(geometry=population_4326, properties={}), f)
    f.flush()

pop_cd = gps.cost_distance(
    friction_layer=road_friction,
    geometries=population_centers, 
    max_distance=1400000.0
)

pop_pp = pop_cd.pyramid()

con_cd = gps.cost_distance(
    friction_layer=road_friction,
    geometries=conflict_centers, 
    max_distance=1400000.0
)

con_pp = con_cd.pyramid()

# prepare color map for weighted overlay based on max cost
breaks = [x for x in range(0, 1000000, 10000)]
colors = gps.get_colors_from_matplotlib(ramp_name='viridis', num_colors=len(breaks))
wo_cm = gps.ColorMap.from_colors(breaks=breaks, color_list=colors)

# our weighted layer avoids conflict centers focusing on just population centers
weighted_overlay = (con_pp * 0.0) + (pop_pp * 1.0)

server = gps.TMS.build(source=weighted_overlay, display=wo_cm)
M.add_layer(TMSRasterData(server), name="WO")
M.add_layer(VectorData("/tmp/population-4326.geojson"), name="Population")

# remove the next to last layer
M.remove_layer(M.layers[-1])
M.remove_layer(M.layers[0])

