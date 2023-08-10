import fiona
import shapely
from shapely.geometry import shape
import pyproj
from functools import partial

projectionWorldGeodetic = pyproj.Proj(init='epsg:4326')
projectionStatePlane = pyproj.Proj(init='epsg:2263', preserve_units=True)
projectionFunction = partial(pyproj.transform, projectionStatePlane, projectionWorldGeodetic)

def getBBL(borough, block, lot):
    BOROUGH_MAP = {
        "MN" : 1,
        "BX" : 2,
        "BK" : 3,
        "QN" : 4,
        "SI" : 5
    }
    return "%d%05d%04d" % (BOROUGH_MAP[borough], block, lot)

fns = [
    "data/nyc/shapefiles/bk_mappluto_16v2/BKMapPLUTO.shp",    
    "data/nyc/shapefiles/bx_mappluto_16v2/BXMapPLUTO.shp",    
    "data/nyc/shapefiles/mn_mappluto_16v2/MNMapPLUTO.shp",    
    "data/nyc/shapefiles/qn_mappluto_16v2/QNMapPLUTO.shp",    
    "data/nyc/shapefiles/si_mappluto_16v2/SIMapPLUTO.shp"
]

bblCentroids = {}
for fn in fns:
    print fn
    f = fiona.open(fn,"r")
    for poly in f:
        geo = shape(poly["geometry"])
        props = poly["properties"]

        bbl = getBBL(props["Borough"], props["Block"], props["Lot"])
        pt = projectionFunction(geo.centroid.x, geo.centroid.y)
        bblCentroids[bbl] = pt
    f.close()

f = open("output/nyc/centroidList.csv","w")
for k,v in bblCentroids.items():
    f.write("%s,%f,%f\n" % (k, v[0], v[1]))
f.close()

print len(bblCentroids)

