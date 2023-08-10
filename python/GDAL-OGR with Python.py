from geographiclib.geodesic import Geodesic

lat1,lon1 = (40.7143528, -74.0059731)  # New York, NY
lat2,lon2 = (1.359, 103.989)   # Delhi, India
g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
g

gc = [Geodesic.WGS84.Direct(lat1, lon1, g['azi1'], i) for i in range(0,int(g['s12']),100000)]

from osgeo import ogr
geomcol =  ogr.Geometry(ogr.wkbGeometryCollection)

point1 = ogr.Geometry(ogr.wkbPoint)
point1.AddPoint(lon1,lat1)
geomcol.AddGeometry(point1)

point2 = ogr.Geometry(ogr.wkbPoint)
point2.AddPoint(lon2,lat2)
geomcol.AddGeometry(point2)

line = ogr.Geometry(ogr.wkbLineString)

[line.AddPoint(i['lon2'],i['lat2']) for i in gc]
geomcol.AddGeometry(line)

data = geomcol.ExportToJson()

get_ipython().system("echo '{data}' > /tmp/geojson.geojson")

#!gist -p /tmp/geojson.geojson

from CesiumWidget import CesiumWidget

cesiumExample = CesiumWidget(width="100%",geojson=data, enable_lighting=True)

cesiumExample

