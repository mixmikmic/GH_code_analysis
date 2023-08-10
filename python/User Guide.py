import telluric as tl
from telluric.constants import WGS84_CRS, WEB_MERCATOR_CRS

gv1 = tl.GeoVector.from_bounds(
    xmin=0, ymin=40, xmax=1, ymax=41, crs=WGS84_CRS
)
print(gv1)

gv1

from shapely.geometry import Polygon

gv2 = tl.GeoVector(
    Polygon([(0, 40), (1, 40.1), (1, 41), (-0.5, 40.5), (0, 40)]),
    WGS84_CRS
)
print(gv2)

print(gv1.centroid)

gv1.area  # Real area in square meters

gv1.is_valid

gv1.within(gv2)

gv1.difference(gv2)

gf1 = tl.GeoFeature(
    gv1,
    {'name': 'One feature'}
)
gf2 = tl.GeoFeature(
    gv2,
    {'name': 'Another feature'}
)
print(gf1)
print(gf2)

fc = tl.FeatureCollection([gf1, gf2])
fc

print(fc.convex_hull)

print(fc.envelope)

fc.save("test_fc.shp")

get_ipython().system('ls test_fc*')

fc.save("test_fc.json")

get_ipython().system('python -m json.tool < test_fc.json | head -n28')

print(list(tl.FileCollection.open("test_fc.shp")))

# This will only save the URL in memory
rs = tl.GeoRaster2.open(
    "https://github.com/mapbox/rasterio/raw/master/tests/data/rgb_deflate.tif"
)

# These calls will fecth some GeoTIFF metadata
# without reading the whole image
print(rs.crs)
print(rs.footprint())
print(rs.band_names)

rs

rs.shape

rs.crop(rs.footprint().buffer(-50000))

rs[200:300, 200:240]

rs[200:300, 200:240].save("test_raster.tif")

