import arcgis
from arcgis.gis import GIS
from IPython.display import display

gis = GIS()

items = gis.content.search("Landsat 8 Views", item_type="Imagery Layer", max_items=2)

for item in items:
    display(item)

l8_views = gis.content.get('4ca13f0e4e29403fa68c46d188c4be73')
l8_views

l8_views.layers

l8_lyr = l8_views.layers[0]
l8_lyr

img_svc_url = 'https://landsat2.arcgis.com/arcgis/rest/services/Landsat8_Views/ImageServer'

from arcgis.raster import ImageryLayer

landsat_lyr = ImageryLayer(img_svc_url)

landsat_lyr

portal = GIS("portal url", "username","password", verify_cert=False)

secure_url = 'https://dev003248.esri.com:6443/arcgis/rest/services/ImgSrv_Landast_Montana2015/ImageServer'

secure_img_lyr = ImageryLayer(secure_url, portal)

secure_img_lyr.url

landsat_lyr.properties.name

landsat_lyr.properties['description']

landsat_lyr.properties.capabilities

landsat_lyr.properties.allowedMosaicMethods

for fn in landsat_lyr.properties.rasterFunctionInfos:
    print(fn['name'])

map = gis.map("Pallikaranai", zoomlevel=13)
map

map.add_layer(landsat_lyr)

import time
from arcgis.raster.functions import apply

for fn in landsat_lyr.properties.rasterFunctionInfos:
    print(fn['name'])
    map.remove_layers()
    map.add_layer(apply(landsat_lyr, fn['name']))
    time.sleep(2)

savi_map = gis.map("Cairo", zoomlevel=6)
savi_map

from arcgis.raster.functions import savi

savi_map.add_layer(savi(landsat_lyr, band_indexes="5 4 0.3"))

from arcgis.raster.functions import *

land_water = stretch(extract_band(landsat_lyr, [4, 5, 3]),
                     stretch_type='PercentClip',
                     min_percent=2, 
                     max_percent=2,
                     dra=True, 
                     gamma=[1, 1, 1])

map2 = gis.map("Pallikaranai", zoomlevel=13)
map2

map2.add_layer(land_water)

