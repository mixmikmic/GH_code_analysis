from osgeo import ogr
import os

shapefile = "../points.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shapefile, 0)
layer = dataSource.GetLayer()

for feature in layer:
    geom = feature.GetGeometryRef()
    print geom.Centroid().ExportToWkt()

from rasterstats import point_query

point = "POINT (725233.602128884 5664901.89211738)"
point_query(point, "../NDVI_example_1.tif")

#from osgeo import ogr
#import os

shapefile = "D:/rasterStats/points.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
dataSource = driver.Open(shapefile, 0)
layer = dataSource.GetLayer()

for feature in layer:
    geom = feature.GetGeometryRef()
    point = geom.Centroid().ExportToWkt()
    print point_query(point, "..NDVI_example_1.tif")

from rasterstats import zonal_stats
stats = zonal_stats("../polys.shp",  "../NDVI_example_1.tif")
print stats[1].keys()
#['count', 'min', 'max', 'mean']
print [f['mean'] for f in stats]
print [f['max'] for f in stats]

for root, dirs, files in os.walk("../"):
    for file in files:
        if file.endswith(".tif"):
            raster = (os.path.join(root, file))
            stats = zonal_stats("../polys.shp",  raster)
            print "filename: ", file
            print [f['mean'] for f in stats]



