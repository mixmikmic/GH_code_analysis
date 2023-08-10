get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np

def bgr_to_rgb(bgr):
    """
    Converts Blue, Green, Red to Red, Green, Blue
    """
    return bgr[..., [2, 1, 0]]

WMS_INSTANCE = ''

sentinel_hub_wms='https://services.sentinel-hub.com/ogc/wms/'+WMS_INSTANCE

layers = {'tulip_field_2016':'ttl1904', 'tulip_field_2017':'ttl1905', 'arable_land_2017':'ttl1917'}

from ipyleaflet import Map, WMSLayer

zoom_level = 13

import math
earth_radius = 6372.7982e3
pixel_size = 2* math.pi * earth_radius * math.cos(math.radians(52.9))/2**(zoom_level+8)
print('Pixel dimension at zoom level %d equals %1.2f m.'%(zoom_level,pixel_size))

m = Map(center=[52.9255665659715, 4.754333496093751], zoom=zoom_level, layout=dict(width='512px', height='512px')); m

s2_layer = WMSLayer(url='https://services.sentinel-hub.com/v1/wms/'+WMS_INSTANCE, layers='TRUE_COLOR', tile_size=512)

m.add_layer(s2_layer)

tulips = WMSLayer(url='http://service.geopedia.world/wms/ml_aws', layers=layers['tulip_field_2017'], tile_size=512, format='image/png', version='1.3.0', TRANSPARENT=True, opacity=0.4)

m.add_layer(tulips)

m.remove_layer(tulips)

import pyproj

def to_epsg3857(latlong_wgs84):
    epsg3857 = pyproj.Proj(init='epsg:3857')
    wgs84    = pyproj.Proj(init='EPSG:4326')
    return pyproj.transform(wgs84,epsg3857,latlong_wgs84[1],latlong_wgs84[0])

bbox_3857 = [to_epsg3857(point) for point in m.bounds]; 

import sys

PATH = '../DataRequest'
sys.path.append(PATH)

from DataRequest import TulipFieldRequest

tulipFields = TulipFieldRequest(bbox=bbox_3857,width=512,height=512,crs=3857,layer=layers['tulip_field_2016'])

tulip_field = tulipFields.get_data()

plt.figure(figsize=(8,8))
plt.imshow(tulip_field[0])



