# GeoPySpark has lots of imports:
from pyspark import SparkContext
import geopyspark as gps

conf=gps.geopyspark_conf(appName="BristleConePine")
conf.set('spark.ui.enabled', True)
sc = SparkContext(conf = conf)

elev_rdd = gps.geotiff.get(
    layer_type='spatial', 
    uri='s3://geopyspark-demo/elevation/ca-elevation.tif')

elev_rdd.count()

elev_tiled_rdd = elev_rdd.tile_to_layout(
    layout=gps.GlobalLayout(), 
    target_crs=3857)
elev_pyramided_rdd = elev_tiled_rdd.pyramid().cache()

from geopyspark.geotrellis.color import ColorMap
from geopyspark.geotrellis.tms import TMSServer
from geonotebook.wrappers import TMSRasterData

from geopyspark.geotrellis.color import get_colors_from_matplotlib
elev_histo        = elev_pyramided_rdd.get_histogram()
elev_colors       = get_colors_from_matplotlib('viridis', 100)
elev_color_map    = gps.ColorMap.from_histogram(elev_histo, elev_colors)

elev_tms = gps.TMS.build(elev_pyramided_rdd, elev_color_map)

M.set_center(-118, 38, 6)

from geonotebook.wrappers import TMSRasterData
M.add_layer(TMSRasterData(elev_tms))

M.remove_layer(M.layers[0])

# use: elev_reprojected_rdd
elev_reclass_pre = elev_tiled_rdd.reclassify({1000:2, 2000:2, 3000:2, 4000:1, 5000:2}, int)
elev_reclass_rdd = elev_reclass_pre.reclassify({1:1}, int)
elev_reclass_pyramid_rdd = elev_reclass_rdd.pyramid()

elev_reclass_histo = elev_reclass_pyramid_rdd.get_histogram()

#elev_reclass_color_map = ColorMap.from_histogram(sc, elev_reclass_histo, get_breaks(sc, 'Viridis', num_colors=100))
elev_reclass_color_map = gps.ColorMap.from_colors(
    breaks =[1], 
    color_list = [0xff000080])

elev_reclass_tms = gps.TMS.build(elev_reclass_pyramid_rdd, elev_reclass_color_map)

M.add_layer(TMSRasterData(elev_reclass_tms))

M.remove_layer(M.layers[0])

from geopyspark.geotrellis.neighborhood import Square
from geopyspark.geotrellis.constants import Operation, Neighborhood

elev_tiled_rdd.srdd.focal(
    Operation.ASPECT.value, 
    'square', 1.0, 0.0, 0.0).rdd().count()

# square_neighborhood = Square(extent=1)
aspect_rdd = elev_tiled_rdd.focal(
    gps.Operation.SLOPE, 
    gps.Neighborhood.SQUARE, 1)

aspect_pyramid_rdd       = aspect_rdd.pyramid()

aspect_histo        = aspect_pyramid_rdd.get_histogram()
aspect_color_map    = gps.ColorMap.from_histogram(aspect_histo, get_colors_from_matplotlib('viridis', num_colors=256))
aspect_tms          = gps.TMS.build(aspect_pyramid_rdd, aspect_color_map)

M.add_layer(TMSRasterData(aspect_tms))

M.remove_layer(M.layers[0])

aspect_reclass_pre  = aspect_rdd.reclassify({120:2, 240:1, 360: 2}, int)
aspect_reclass      = aspect_reclass_pre.reclassify({1:1}, int)

aspect_reclass_pyramid_rdd       = aspect_reclass.pyramid()

aspect_reclass_histo       = aspect_reclass_pyramid_rdd.get_histogram()
aspect_reclass_color_map   = gps.ColorMap.from_histogram(aspect_reclass_histo, get_colors_from_matplotlib('viridis', num_colors=256))
aspect_reclass_tms         = gps.TMS.build(aspect_reclass_pyramid_rdd, aspect_reclass_color_map)

M.add_layer(TMSRasterData(aspect_reclass_tms))

M.remove_layer(M.layers[0])

added = elev_reclass_pyramid_rdd + aspect_reclass_pyramid_rdd

added_histo = added.get_histogram()
added_color_map = gps.ColorMap.from_histogram(added_histo, get_colors_from_matplotlib('viridis', num_colors=256))
added_tms = gps.TMS.build(added, added_color_map)

M.add_layer(TMSRasterData(added_tms))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

v = elev_tiled_rdd.lookup(342,787)
plt.imshow(v[0]['data'][0])

