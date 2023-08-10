# connect to the GIS
from arcgis.gis import GIS
gis = GIS("https://python.playground.esri.com/portal", "arcgis_python", "amazing_arcgis_123")

gis.content.create_folder('packages')

tpk_item = gis.content.add({}, data='data/USA_counties_divorce_rate.tpk', folder='packages')
tpk_item

tile_layer = tpk_item.publish()
tile_layer

# upload vector tile package to the portal
vtpk_item = gis.content.add({}, data='data/World_earthquakes_2010.vtpk', folder='packages')
vtpk_item

# publish that item as a vector tile layer
vtpk_layer = vtpk_item.publish()

vtpk_layer

slpk_item = gis.content.add({}, data='data/World_earthquakes_2000_2010.slpk', folder='packages')
slpk_item

slpk_layer = slpk_item.publish()
slpk_layer

