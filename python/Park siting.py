import geopyspark as gps
import fiona

from pyspark import SparkContext, StorageLevel
from shapely.geometry import MultiPoint, MultiPolygon, shape
from geonotebook.wrappers import VectorData, TMSRasterData

conf = gps.geopyspark_conf(appName="park-siting", master="local[*]")
sc = SparkContext.getOrCreate(conf=conf)

M.set_center(-122.1, 37.75, 10)

get_ipython().system('curl -o /tmp/bart.geojson https://s3.amazonaws.com/geopyspark-demo/bayarea/bart.geojson')
get_ipython().system('curl -o /tmp/school.geojson https://s3.amazonaws.com/geopyspark-demo/bayarea/school.geojson')
get_ipython().system('curl -o /tmp/parks.geojson https://s3.amazonaws.com/geopyspark-demo/bayarea/parks.geojson')

with fiona.open("/tmp/bart.geojson") as source:
    bart_crs = source.crs['init']
    bart = MultiPoint([shape(f['geometry']) for f in source])

with fiona.open("/tmp/school.geojson") as source:
    schools_crs = source.crs['init']
    schools = MultiPoint([shape(f['geometry']) for f in source])

with fiona.open("/tmp/parks.geojson") as source:
    parks_crs = source.crs['init']
    parks = MultiPolygon([shape(f['geometry']) for f in source])

bart_layer = gps.euclidean_distance(geometry=bart,
                                    source_crs=bart_crs,
                                    zoom=12)

schools_layer = gps.euclidean_distance(geometry=schools,
                                       source_crs=schools_crs,
                                       zoom=12)

parks_layer = gps.euclidean_distance(geometry=parks,
                                     source_crs=parks_crs,
                                     zoom=12)

# Persists each layer to memory and disk
bart_layer.persist(StorageLevel.MEMORY_AND_DISK)
schools_layer.persist(StorageLevel.MEMORY_AND_DISK)
parks_layer.persist(StorageLevel.MEMORY_AND_DISK)

weighted_layer = -1 * bart_layer - schools_layer + 3 * parks_layer

# Persists the weighted layer to memory and disk
weighted_layer.persist(StorageLevel.MEMORY_AND_DISK)

# The following code may take awhile to complete
reprojected = weighted_layer.tile_to_layout(layout=gps.GlobalLayout(),
                                            target_crs="EPSG:3857")
pyramid = reprojected.pyramid()
histogram = pyramid.get_histogram()

color_map = gps.ColorMap.build(breaks=histogram,
                               colors='viridis')

tms = gps.TMS.build(source=pyramid,
                    display=color_map)

M.add_layer(TMSRasterData(tms))
M.add_layer(VectorData("/tmp/bart.geojson"), name="BART stops")
M.add_layer(VectorData("/tmp/parks.geojson"), name="Parks")

M.remove_layer(M.layers[2])
M.remove_layer(M.layers[1])
M.remove_layer(M.layers[0])

