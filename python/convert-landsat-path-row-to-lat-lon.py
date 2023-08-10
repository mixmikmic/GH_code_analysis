import ogr
import shapely.wkt
import shapely.geometry
import urllib
import zipfile

url = "https://landsat.usgs.gov/sites/default/files/documents/wrs2_asc_desc.zip"
filehandle, _ = urllib.urlretrieve(url)
zip_file_object = zipfile.ZipFile(filehandle, 'r')
zip_file_object.extractall(".")
zip_file_object.close()

shapefile = 'wrs2_asc_desc/wrs2_asc_desc.shp'
wrs = ogr.Open(shapefile)
layer = wrs.GetLayer(0)

lon = -105.2705
lat = 40.0150
point = shapely.geometry.Point(lon, lat)
mode = 'D'

def checkPoint(feature, point, mode):
    geom = feature.GetGeometryRef() #Get geometry from feature
    shape = shapely.wkt.loads(geom.ExportToWkt()) #Import geometry into shapely to easily work with our point
    if point.within(shape) and feature['MODE']==mode:
        return True
    else:
        return False

i=0
while not checkPoint(layer.GetFeature(i), point, mode):
    i += 1
feature = layer.GetFeature(i)
path = feature['PATH']
row = feature['ROW']
print('Path: ', path, 'Row: ', row)

