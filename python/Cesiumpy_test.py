import cesiumpy
v = cesiumpy.Viewer()
v.camera.flyTo("La Push, WA")
v

# Set viewpoint as (longitude,latitude,elevation)
from geopy.geocoders import Nominatim
geolocator = Nominatim()
location = geolocator.geocode("La Push, WA")
elevation = 5000  # meters 
long_lat_elev = (location.longitude, location.latitude, elevation)

v = cesiumpy.Viewer()
v.camera.flyTo(long_lat_elev)
v

import cesiumpy
ds = cesiumpy.KmlDataSource('./gauges.kml')

if 0:
    v = cesiumpy.Viewer()
    v.dataSources.add(ds)
    v.camera.flyTo(long_lat_elev)
    v



