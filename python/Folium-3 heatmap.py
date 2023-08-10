from bs4 import BeautifulSoup
import urllib
import folium
from folium import plugins
from functions import *

# let's get the earthquake larger than M5 globally from 2010-01-01 to 2016-01-01. 
url = build_query(outFormat = 'text', starttime = '2010-01-01', endtime = '2016-06-01', minmagnitude = 5.0)

# get the earthquake data from USGS and parse them into a numpy array
r = urllib.urlopen(url).read()
soup = BeautifulSoup(r, "lxml")
events_mat = parse_result(soup.text)

# extract lat, lon, and magnitude for the folium heatmap
lats = [float(item[2]) for item in events_mat]
lons = [float(item[3]) for item in events_mat]
mag = [float(item[4]) for item in events_mat]

# Using USGS style tile
url_base = 'http://server.arcgisonline.com/ArcGIS/rest/services/'
service = 'NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
tileset = url_base + service

m = folium.Map(location=[37.8716, -122.2727], zoom_start=2,                control_scale = True, tiles=tileset, attr='USGS style')

# I am using the magnitude as the weight for the heatmap
m.add_children(plugins.HeatMap(zip(lats, lons, mag), radius = 10))



