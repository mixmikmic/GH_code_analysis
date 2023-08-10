import rasterio
import rasterio.features
import rasterio.warp
import geopyspark as gps
import numpy as np
import csv
import matplotlib.pyplot as plt

from datetime import datetime
from pyspark import SparkContext
from geonotebook.wrappers import TMSRasterData
from osgeo import osr

get_ipython().magic('matplotlib inline')

sc = SparkContext(conf=gps.geopyspark_conf(appName="Landsat"))

csv_data = []
with open("/tmp/l8-scenes.csv") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        csv_data.append(row)

rdd0 = sc.parallelize(csv_data)

def get_metadata(line):
    
    try:
        with rasterio.open(line['uri']) as dataset:
            bounds = dataset.bounds
            height = height = dataset.height
            width = dataset.width
            crs = dataset.get_crs()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs.wkt)
            proj4 = srs.ExportToProj4()
            ws = [w for (ij, w) in dataset.block_windows()]
    except:
            ws = []
            
    def windows(line, ws):
        for w in ws:
            ((row_start, row_stop), (col_start, col_stop)) = w

            left  = bounds.left + (bounds.right - bounds.left)*(float(col_start)/width)
            right = bounds.left + (bounds.right - bounds.left)*(float(col_stop)/ width)
            bottom = bounds.top + (bounds.bottom - bounds.top)*(float(row_stop)/height)
            top = bounds.top + (bounds.bottom - bounds.top)*(float(row_start)/height)
            extent = gps.Extent(left,bottom,right,top)
            instant = datetime.strptime(line['date'], '%Y-%m-%d')
                
            new_line = line.copy()
            new_line.pop('date')
            new_line.pop('scene_id')
            new_line['window'] = w
            new_line['projected_extent'] = gps.TemporalProjectedExtent(extent=extent, instant=instant, proj4=proj4)
            yield new_line
    
    return [i for i in windows(line, ws)]

rdd1 = rdd0.flatMap(get_metadata)
rdd1.first()

def get_data(line):
    
    new_line = line.copy()

    with rasterio.open(line['uri']) as dataset:
        new_line['data'] = dataset.read(1, window=line['window'])
        new_line.pop('window')
        new_line.pop('uri')
    
    return new_line

rdd2 = rdd1.map(get_data)
rdd2.first()

rdd3 = rdd2.groupBy(lambda line: line['projected_extent'])
rdd3.first()

def make_tiles(line):
    projected_extent = line[0]
    array = np.array([l['data'] for l in sorted(line[1], key=lambda l: l['band'])])
    tile = gps.Tile.from_numpy_array(array, no_data_value=0)
    return (projected_extent, tile)

def interesting_tile(line):
    [tpe, tile] = line
    return (np.sum(tile[0][0]) != 0)

def square_tile(line):
    [tpe, tile] = line
    return tile[0][0].shape == (512,512)

rdd4 = rdd3.map(make_tiles).filter(square_tile)
data = rdd4.filter(interesting_tile).first()
data

plt.imshow(data[1][0][0])

raster_layer = gps.RasterLayer.from_numpy_rdd(gps.LayerType.SPACETIME, rdd4)

tiled_raster_layer = raster_layer.tile_to_layout(layout = gps.GlobalLayout(), target_crs=3857)

pyramid = tiled_raster_layer.pyramid()

for layer in pyramid.levels.values():
    gps.write("file:///tmp/catalog/", "landsat", layer, time_unit=gps.TimeUnit.DAYS)

pyramid = tiled_raster_layer.to_spatial_layer().pyramid()

colormap = gps.ColorMap.build(breaks=tiled_raster_layer.get_histogram(), colors='plasma')

tms_server = gps.TMS.build(pyramid, display=colormap)

M.add_layer(TMSRasterData(tms_server), name="landsat")

M.remove_layer(M.layers[0])



