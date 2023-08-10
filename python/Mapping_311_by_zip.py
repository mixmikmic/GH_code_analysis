import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

#Read the csv file that was downloaded from the City of Austin: "https://data.austintexas.gov/"
Austin_311_df = pd.read_csv('data/All_Austin_311.csv',low_memory=False)

Austin_311_df.head()

#Use the 'clean_Austin_311_df' and keep the approriate columns for this analysis.

austin_lat_lon = Austin_311_df[["incident_zip", "owning_department","latitude", "longitude"]]

# Change the zip to an integer
austin_lat_lon["incident_zip"] = austin_lat_lon["incident_zip"].astype(int)
austin_lat_lon = austin_lat_lon.rename(columns={"incident_zip":"Zip Code","owning_department":"Number of Complaints",
                                              "latitude": "Lat", "longitude":"Lon"})

# Get the number of complaints from the "Department" and means of the Lat and Lon.
atx_zip_by_latlon = austin_lat_lon.groupby("Zip Code").agg({"Number of Complaints": 'count', "Lat": 'mean', "Lon": 'mean'})
atx_zip_by_latlon.head()

total = atx_zip_by_latlon['Number of Complaints'].sum()
print(total)

atx_zip_by_latlon["Coordinates"] = atx_zip_by_latlon[['Lat', 'Lon']].apply(tuple, axis=1)
atx_zip_by_latlon.head()

#Plotting the 311 call density on Travis county zip codes marklines.

fig, ax = plt.subplots(figsize=(10,10))
map = Basemap(resolution='f', # c, l, i, h, f or None
            projection='merc',
            lat_0=30.3, lon_0=-97.8,
            llcrnrlon=-98.2, llcrnrlat=30.02, urcrnrlon=-97.2, urcrnrlat=30.6) 

map.drawmapboundary(fill_color="#FFFFFF")
map.drawrivers(color="#0000FF")

map.readshapefile('USzip_codes/cb_2016_us_zcta510_500k', 'cb_2016_us_zcta510_500k')

def plot_area(Coordinates):
    count = atx_zip_by_latlon.loc[atx_zip_by_latlon.Coordinates == Coordinates]["Number of Complaints"]
    x, y = map(Coordinates[1], Coordinates[0])
    #The count number is too high to plot. So reduce it and plot the area of a circle.
    size = (count/6000) ** 2 + 3.14
    map.plot(x, y, 'o', markersize=size, color='#CC5500', alpha=0.8)


atx_zip_by_latlon.Coordinates.apply(plot_area)
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('images/311_call_density_zip.png', dpi=200)

# Import necessary modules
import gmaps
import json
import gmaps.geojson_geometries
import gmaps.datasets
from config import g_key
gmaps.configure(g_key)

#Draw the Google heatmap using Lat and Lon from 'atx_zip_by_latlon'
with open("travis.geojson") as f:
    geometry = json.load(f)
    
fig = gmaps.figure()
heatmap_layer = gmaps.heatmap_layer(
    atx_zip_by_latlon[["Lat", "Lon"]], weights=atx_zip_by_latlon["Number of Complaints"],
    max_intensity=50, point_radius=7.5)

geojson_layer = gmaps.geojson_layer(geometry, fill_color=None , fill_opacity=0.0)
fig.add_layer(heatmap_layer)
fig.add_layer(geojson_layer)
fig

austin_lat_lon['Count'] = (austin_lat_lon['Number of Complaints'] !='').astype(int)
    
austin_lat_lon.head()

#Draw the Google heatmap using Lat and Lon from 'atx_zip_by_latlon'
with open("travis.geojson") as f:
    geometry = json.load(f)
    
fig = gmaps.figure()
heatmap_layer = gmaps.heatmap_layer(
    austin_lat_lon[["Lat", "Lon"]], weights=austin_lat_lon["Count"],
    max_intensity=10, point_radius=0.5)

geojson_layer = gmaps.geojson_layer(geometry, fill_color=None , fill_opacity=0.0)
fig.add_layer(heatmap_layer)
fig.add_layer(geojson_layer)
fig



