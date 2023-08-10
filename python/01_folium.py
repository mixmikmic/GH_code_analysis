import numpy as np 
import pandas as pd
import folium

bj_aq = pd.read_csv("../data/bj_201803_renamed_aq.csv")

stations = pd.read_csv("../data/Beijing_AirQuality_Stations.csv")
stations.head()

map_hooray = folium.Map(
    location=[39.929, 116.417],
    tiles = "Stamen Terrain",
    zoom_start = 10) # Uses lat then lon. The bigger the zoom number, the closer in you get
map_hooray # Calls the map to display 

data = pd.merge(bj_aq[bj_aq["time"]=="2017-01-01 14:00:00"], stations, on="station_id")
data.head()

map_hooray = folium.Map(
    location=[39.929, 116.8],
    tiles = "Stamen Terrain",
    zoom_start = 9) 

for _, row in data.iterrows():
    folium.Marker([
        row["Latitude"],row["Longitude"]],
        popup=row["station_id"]).add_to(map_hooray)

map_hooray

def colourgrad(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = (value-minimum) / (maximum - minimum)
    g = int(max(0, 128*(1 - ratio)))
    r = int(max(0, 128*(1 - ratio) + 127))
    b = 0
    hexcolour = '#%02x%02x%02x' % (r,g,b)
    return hexcolour

data = pd.merge(bj_aq[bj_aq["time"]=="2017-01-01 15:00:00"], stations, on="station_id")
map_hooray = folium.Map(
    location=[39.929, 116.8],
    tiles = "Stamen Terrain",
    zoom_start = 9) 

for _, row in data.iterrows():
    color = colourgrad(0, 500, min(row["PM25_Concentration"], 500))
    folium.CircleMarker([
        row["Latitude"],row["Longitude"]],
        color=color, radius=9, fill_opacity=1, fill=True, fill_color=color,
        popup=row["station_id"] + ":" + str(row["PM25_Concentration"])).add_to(map_hooray)

# CWD = os.getcwd()
# map_ws.save('osm.html')
# webbrowser.open_new('file://'+CWD+'/'+'osm.html')
map_hooray

data = pd.merge(bj_aq[bj_aq["time"]=="2017-01-01 18:00:00"], stations, on="station_id")
map_hooray = folium.Map(
    location=[39.929, 116.8],
    tiles = "Stamen Terrain",
    zoom_start = 9) 

for _, row in data.iterrows():
    color = colourgrad(0, 500, min(row["PM25_Concentration"], 500))
    folium.CircleMarker([
        row["Latitude"],row["Longitude"]],
        color=color, radius=9, fill_opacity=1, fill=True, fill_color=color,
        popup=row["station_id"] + ":" + str(row["PM25_Concentration"])).add_to(map_hooray)

map_hooray

data = pd.merge(bj_aq[bj_aq["time"]=="2017-01-01 20:00:00"], stations, on="station_id")
map_hooray = folium.Map(
    location=[39.929, 116.8],
    tiles = "Stamen Terrain",
    zoom_start = 9) 

for _, row in data.iterrows():
    color = colourgrad(0, 500, min(row["PM25_Concentration"], 500))
    folium.CircleMarker([
        row["Latitude"],row["Longitude"]],
        color=color, radius=9, fill_opacity=1, fill=True, fill_color=color,
        popup=row["station_id"] + ":" + str(row["PM25_Concentration"])).add_to(map_hooray)

map_hooray

