import folium

# Add the USGS style tile
url_base = 'http://server.arcgisonline.com/ArcGIS/rest/services/'
service = 'NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
tileset = url_base + service

map_1 = folium.Map(location=[37.8716, -122.2727], zoom_start=10,                      control_scale = True, tiles=tileset, attr='USGS style')

map_1.add_children(folium.Marker([37.8716, -122.2727], popup = 'I am here'))
map_1

