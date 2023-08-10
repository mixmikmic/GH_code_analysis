from geopyspark import geopyspark_conf
from geopyspark.geotrellis.constants import SPATIAL, NODATAINT, MAX, SQUARE, EXACT
from geopyspark.geotrellis.layer import TiledRasterLayer
from pyspark import SparkContext
import json
import shapely
from shapely.geometry import shape, asShape, MultiPoint, MultiLineString

conf = geopyspark_conf("local[*]", "TMS")
pysc = SparkContext.getOrCreate(conf)

from geopyspark.geotrellis.tms import *
from geopyspark.geotrellis.color import ColorMap
from geonotebook.wrappers.raster import TMSRasterData

nlcd_layer_name = "nlcd-tms-epsg3857"

nlcd = s3_catalog_tms_server(
    pysc, 
    "azavea-datahub", "catalog", 
    nlcd_layer_name, 
    ColorMap.nlcd_colormap(pysc))

M.add_layer(TMSRasterData(nlcd), name="nlcd")

M.set_center(-85.2934168635424, 35.02445474101138, 9)

get_ipython().system('curl -o /tmp/mask.json https://s3.amazonaws.com/chattademo/chatta_mask.json')

from geonotebook.wrappers import VectorData
vd = VectorData("/tmp/mask.json")
name = "Outline"
M.add_layer(vd, name=name)

from functools import partial
import fiona
import json
import pyproj
from shapely.geometry import mapping, shape
from shapely.ops import transform

project = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:4326'),
    pyproj.Proj(init='epsg:3857'))

txt = open('/tmp/mask.json').read()
js = json.loads(txt)
geom = shape(js)
center = geom.centroid
chatta_poly = transform(project, geom)
chatta_poly

from geopyspark.geotrellis import catalog

MAX_ZOOM = 12
query_rdd = catalog.query(
    geopysc, SPATIAL, 
    "s3://azavea-datahub/catalog", nlcd_layer_name, 
    MAX_ZOOM, intersects=chatta_poly)

chatta_rdd = query_rdd.convert_data_type("int8").cache()

chatta_rdd.get_min_max()

chatta_rdd.layer_metadata.extent

chatta_py_rdd = chatta_rdd.to_numpy_rdd()

