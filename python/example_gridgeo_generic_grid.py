import gridgeo

gridgeo.__version__

url = 'http://thredds.cencoos.org/thredds/dodsC/CA_DAS.nc'

grid = gridgeo.GridGeo(url)
grid

grid.__geo_interface__.keys()

grid.outline

print('There are {} polygons.'.format(len(grid.polygons)))

grid.polygons[0:10]

img = grid.raster

type(img)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(img)
_ = ax.axis('off')

grid.grid

grid.mesh

grid.nc

properties = dict(fill='#fd7d11',
                  fill_opacity=0.2,
                  stroke_opacity=1)

geojson = grid.to_geojson(**properties)

geojson['properties']

import json

kw = dict(sort_keys=True, indent=4, separators=(',', ': '))
with open('grid.geojson','w') as f:
    json.dump(geojson, f, **kw)

import fiona

schema = {'geometry': 'MultiPolygon',
          'properties': {'name': 'str:{}'.format(len(grid.mesh))}}

with fiona.open('grid.shp', 'w', 'ESRI Shapefile', schema) as f:
    f.write({'geometry': grid.__geo_interface__,
             'properties': {'name': grid.mesh}})

import folium

x, y = grid.outline.centroid.xy

mapa = folium.Map(location=[y[0], x[0]])

folium.GeoJson(grid.outline.__geo_interface__).add_to(mapa)

min_lon, min_lat, max_lon, max_lat = grid.outline.bounds
mapa.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
mapa

