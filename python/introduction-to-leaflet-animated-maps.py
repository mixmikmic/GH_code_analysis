get_ipython().system('pip install folium')
get_ipython().system('pip install pandas')

import folium
from folium.plugins import MarkerCluster
import pandas as pd

#Define coordinates of where we want to center our map
boulder_coords = [40.015, -105.2705]

#Create the map
my_map = folium.Map(location = boulder_coords, zoom_start = 13)

#Display the map
my_map

#Define the coordinates we want our markers to be at
CU_Boulder_coords = [40.007153, -105.266930]
East_Campus_coords = [40.013501, -105.251889]
SEEC_coords = [40.009837, -105.241905]

#Add markers to the map
folium.Marker(CU_Boulder_coords, popup = 'CU Boulder').add_to(my_map)
folium.Marker(East_Campus_coords, popup = 'East Campus').add_to(my_map)
folium.Marker(SEEC_coords, popup = 'SEEC Building').add_to(my_map)

#Display the map
my_map

#Adding a tileset to our map
map_with_tiles = folium.Map(location = boulder_coords, tiles = 'Stamen Toner')
map_with_tiles

#Using polygon markers with colors instead of default markers
polygon_map = folium.Map(location = boulder_coords, zoom_start = 13)

CU_Boulder_coords = [40.007153, -105.266930]
East_Campus_coords = [40.013501, -105.251889]
SEEC_coords = [40.009837, -105.241905]

#Add markers to the map
folium.RegularPolygonMarker(CU_Boulder_coords, popup = 'CU Boulder', fill_color = '#00ff40',
                            number_of_sides = 3, radius = 10).add_to(polygon_map)
folium.RegularPolygonMarker(East_Campus_coords, popup = 'EastCampus', fill_color = '#bf00ff',
                            number_of_sides = 5, radius = 10).add_to(polygon_map)
folium.RegularPolygonMarker(SEEC_coords, popup = 'SEEC Building', fill_color = '#ff0000',
                           number_of_sides = 8, radius = 10).add_to(polygon_map)

#Display the map
polygon_map



