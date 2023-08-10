# Loading neccessary libraries
import folium
import branca
# You can install branca with pip, should work on all platforms
import pandas as pd

# Loading datasets will be using
crimedata = pd.read_csv('input/SFPD_Incidents_-_Previous_Year__2015_.csv')
volcanoes = pd.read_csv("input/Volcanoes_USA.txt")
state_data = pd.read_csv("input/US_Unemployment_Oct2012.csv")

# Simple map centered around particular coordinates
# Map is a main classs here which basically draws the map
map_osm = folium.Map(location=[40., 15.], zoom_start=6)
# By default folium saves the map to html file. In order to have it displayed in notebook just call the instance of a map
map_osm

# One of the key features in folium is marker
# It is basically a point on the specific location of a map
# We create map instance
m = folium.Map()
# We create a Marker with defined location and then we add this Marker to the map
folium.Marker(location=[40., 15.]).add_to(m)
m

# Let's try something more complex
# As we will work with data about crimes in San Francisco, we need to define coordinates of this city
SF_COORDINATES = (37.76, -122.45)

# for speed purposes we will just use 1000 samples from dataset
MAX_RECORDS = 1000

# create empty map zoomed in on San Francisco
map = folium.Map(location=SF_COORDINATES, zoom_start=12)

# add a marker for every record in the filtered data, use a clustered view
fg=folium.FeatureGroup(name="SF Crime Locations")
for each in crimedata[0:MAX_RECORDS].iterrows():
    fg.add_child(folium.Marker(location = [each[1]['Y'],each[1]['X']]))
# Basically we have a marker for each crime happened in SF
map.add_child(fg)
map

# No let's try to make a choropleth map based on the same crime data

# definition of the boundaries in the map
# You see how we use geojson file here
district_geo = r'geo_json/sfpddistricts.geojson'

# calculating total number of incidents per district
crimedata2 = pd.DataFrame(crimedata['PdDistrict'].value_counts().astype(float))
crimedata2 = crimedata2.reset_index()
crimedata2.columns = ['District', 'Number']
 
# creation of the choropleth
map1 = folium.Map(location=SF_COORDINATES, zoom_start=12)
map1.choropleth(geo_path = district_geo,  
              data = crimedata2,
              columns = ['District', 'Number'],
              key_on = 'feature.properties.DISTRICT',
              fill_color = 'YlOrRd', 
              fill_opacity = 0.7, 
              line_opacity = 0.2,
              legend_name = 'Number of incidents per district')
              
map1

map2=folium.Map(location=[volcanoes['LAT'].mean(),volcanoes['LON'].mean()],zoom_start=6,tiles='Mapbox bright')

# Small function which will help us to color markers
def color(elev):
    minimum=int(min(volcanoes['ELEV']))
    step=int((max(volcanoes['ELEV'])-min(volcanoes['ELEV']))/3)
    if elev in range(minimum,minimum+step):
        col='green'
    elif elev in range(minimum+step,minimum+step*2):
        col='orange'
    else:
        col='red'
    return col

fg=folium.FeatureGroup(name="Volcano Locations")
# Here we are creating all markers that correspond to data
for lat,lon,name,elev in zip(volcanoes['LAT'],volcanoes['LON'],volcanoes['NAME'],volcanoes['ELEV']):
    fg.add_child(folium.Marker(location=[lat,lon],
                               popup=(folium.Popup(name)),
                               icon=folium.Icon(color=color(elev),
                                                icon_color='green')))
map2.add_child(fg)
# With the following code notice how we use lambda function to additionally color the markers
map2.add_child(folium.GeoJson(data=open('geo_json/world_geojson_from_ogr.json'),
               name="Population",
               style_function=lambda x: {'fillColor':'green' if x['properties']['POP2005'] <= 10000000 else 'orange' \
                          if 10000000 < x['properties']['POP2005'] < 20000000 else 'red'}))
map2.add_child(folium.LayerControl())
map2

waypoints = folium.Map(location=[46.8527, -121.7649], tiles='Stamen Terrain',
                       zoom_start=13)
folium.Marker([46.8354, -121.7325], popup='Camp Muir').add_to(waypoints)
folium.ClickForMarker().add_to(waypoints)
waypoints

# Again we are using a different geojson file
state_geo = r'geo_json/us-states.json'

# Let Folium determine the scale
states = folium.Map(location=[48, -102], zoom_start=3)
states.choropleth(geo_path=state_geo, data=state_data,
                columns=['State', 'Unemployment'],
                key_on='feature.id',
                fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2,
                legend_name='Unemployment Rate (%)')

states



