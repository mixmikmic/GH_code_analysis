import folium
from folium import plugins
from scipy.ndimage import imread

# boundary of the image on the map
min_lon = -123.5617
max_lon = -121.0617
min_lat = 37.382166
max_lat = 39.048834

# create the map
map_ = folium.Map(location=[38.2, -122],
                  tiles='Stamen Terrain', zoom_start = 8)

# read in png file to numpy array
data = imread('./ii_overlay.png')

Intensity = plugins.ImageOverlay(data, opacity=0.8,         bounds =[[min_lat, min_lon], [max_lat, max_lon]],        attr = 'Shaking Intensity')

# add a name to the layer
Intensity.layer_name = 'Shaking Intensity'

# Overlay the image
map_.add_children(Intensity)

# add the layer control
folium.LayerControl().add_to(map_)

map_

