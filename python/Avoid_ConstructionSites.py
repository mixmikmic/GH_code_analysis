import json
import requests

import folium
import pyproj
from shapely import geometry
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

from openrouteservice import client

url = 'https://geo.sv.rostock.de/download/opendata/baustellen/baustellen.json'

def CreateBufferPolygon(point_in, resolution=10, radius=10):    

    sr_wgs = pyproj.Proj(init='epsg:4326')
    sr_utm = pyproj.Proj(init='epsg:32632') # WGS84 UTM32N
    point_in_proj = pyproj.transform(sr_wgs, sr_utm, *point_in) # unpack list to arguments
    point_buffer_proj = Point(point_in_proj).buffer(radius, resolution=resolution) # 10 m buffer
    
    # Iterate over all points in buffer and build polygon
    poly_wgs = []
    for point in point_buffer_proj.exterior.coords:
        poly_wgs.append(pyproj.transform(sr_utm, sr_wgs, *point)) # Transform back to WGS84

    return poly_wgs
    

# Set up the fundamentals
api_key = '58d904a497c67e00015b45fc40d3503b3a9a4695936156d392dbf0e3' # Individual api key
clnt = client.Client(key=api_key) # Create client with api key
rostock_json = requests.get(url).json() # Get data as JSON

map_params = {'tiles':'Stamen Toner',
              'location':([54.13207, 12.101612]),
              'zoom_start': 12}
map1 = folium.Map(**map_params)

# Populate a construction site buffer polygon list
sites_poly = []
for site_data in rostock_json['features']:
    site_coords = site_data['geometry']['coordinates']
    folium.features.Marker(list(reversed(site_coords)),
                           popup='Construction point<br>{0}'.format(site_coords)).add_to(map1)
    
    # Create buffer polygons around construction sites with 10 m radius and low resolution
    site_poly_coords = CreateBufferPolygon(site_coords,
                                           resolution=2, # low resolution to keep polygons lean
                                           radius=10)
    sites_poly.append(site_poly_coords)
    
    site_poly_coords = [(y,x) for x,y in site_poly_coords] # Reverse coords for folium/Leaflet
    folium.features.PolygonMarker(locations=site_poly_coords,
                                  color='#ffd699',
                                  fill_color='#ffd699',
                                  fill_opacity=0.2,
                                  weight=3).add_to(map1)
    
map1

# GeoJSON style function
def style_function(color):
    return lambda feature: dict(color=color,
                              weight=3,
                              opacity=0.5)

# Create new map to start from scratch
map_params.update({'location': ([54.091389, 12.096686]),
                   'zoom_start': 13})
map2 = folium.Map(**map_params)

# Request normal route between appropriate locations without construction sites
request_params = {'coordinates': [[12.108259, 54.081919],
                                 [12.072063, 54.103684]],
                'format_out': 'geojson',
                'profile': 'driving-car',
                'preference': 'shortest',
                'instructions': 'false',}
route_normal = clnt.directions(**request_params)
folium.features.GeoJson(data=route_normal,
                        name='Route without construction sites',
                        style_function=style_function('#FF0000'),
                        overlay=True).add_to(map2)

# Buffer route with 0.009 degrees (really, just too lazy to project again...)
route_buffer = LineString(route_normal['features'][0]['geometry']['coordinates']).buffer(0.009)
folium.features.GeoJson(data=geometry.mapping(route_buffer),
                        name='Route Buffer',
                        style_function=style_function('#FFFF00'),
                        overlay=True).add_to(map2)

# Plot which construction sites fall into the buffer Polygon
sites_buffer_poly = []
for site_poly in sites_poly:
    poly = Polygon(site_poly)
    if route_buffer.intersects(poly):
        folium.features.Marker(list(reversed(poly.centroid.coords[0]))).add_to(map2)
        sites_buffer_poly.append(poly)

map2

# Add the site polygons to the request parameters
request_params['options'] = {'avoid_polygons': geometry.mapping(MultiPolygon(sites_buffer_poly))}
route_detour = clnt.directions(**request_params)

folium.features.GeoJson(data=route_detour,
                        name='Route with construction sites',
                        style_function=style_function('#00FF00'),
                        overlay=True).add_to(map2)

map2.add_child(folium.map.LayerControl())
map2

