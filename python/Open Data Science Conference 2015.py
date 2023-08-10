import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
tripdir = './data/rk/segments/328635286/'
files = os.listdir(tripdir)
files[:20]

import json
geo = []
for fn in files:
    with open(tripdir + fn, 'r') as f:
        geo.append(json.load(f))
geo[0]

import geopandas as gpd
segments = gpd.GeoDataFrame.from_features(geo)
segments.head()

# Plot using built-in matplotlib plotting
segments.plot()

import geojsonio
geojsonio.embed(segments.to_json())

def load_random_trips(n_trips):
    # Load a random selection of running trips
    trip_df = pd.read_csv('./data/rk/boston_running_trips_201404_201409.tsv', sep='\t')
    idx = random.sample(trip_df.index, n_trips)
    trip_list = trip_df['tripid'][idx].values

    # Find all corresponding segment files
    data_dir = './data/rk/segments/'
    file_list = []
    for tripid in trip_list:
        trip_dir = data_dir + str(tripid) + '/'
        if os.path.isdir(trip_dir):
            file_list += [trip_dir + s for s in os.listdir(trip_dir)]

    n_segments = len(file_list)
    
    # Load files and create GeoDataFrame
    segments = []
    features = []

    for path in file_list:
        f = open(path, 'r')
        geo = json.load(f)
        if (len(geo['geometry']['coordinates']) > 1):  # Make sure LineStrings have at least 2 coordinates    
            features.append(geo)
            segments.append(path.split('/')[-1].split('.')[0])  # Save segment/file identifier
        f.close

    gdf = gpd.GeoDataFrame.from_features(features)  
    gdf.index = segments

    c = 26.8224  # Factor to convert seconds/meter to minutes/mile
    gdf['pace_mpm'] = gdf['duration']/gdf['distance']*c  # Compute pace of each segment in minutes/mile

    return gdf

# Load segments from a random sample of RunKeeper trips
gdf = load_random_trips(10000)

n_segments = len(gdf)
median_pace = gdf['pace_mpm'].median()

print('Number of segments = {:,d}'.format(n_segments))
print('Median segment pace (min/mile) = {:0.2f}'.format(median_pace))

# Visualize a random selection of segments
idx = random.sample(range(len(gdf)), 200)
geo = gdf[['geometry']].iloc[idx].to_json()
geojsonio.embed(geo)

# Load City of Cambridge boundary
shpfile = './data/cambridge/BOUNDARY_CityBoundary.shp'
cambridge = gpd.GeoDataFrame.from_file(shpfile)
cambridge.to_crs(epsg=4326, inplace=True)  # Project local coordinate system into lon/lat

# Now, we can do logical indexing of the GeoDataFrame using spatial operations
boundary = cambridge['geometry'][0]
idx = gdf['geometry'].intersects(boundary)
gdf = gdf[idx]

print('Number of segments = {:,d}'.format(len(gdf)))

# Visualize a random selection of segments; they should now be only in Cambridge
idx = random.sample(range(len(gdf)), 200)
geo = gdf[['geometry']].iloc[idx].to_json()
geojsonio.embed(geo)

# Find top 10 worst intersections (of ones with at least 100 segments)
idx = intersections['segment_count'] >= 100
top_ten = intersections[['geometry']][idx].head(10)
top_ten['marker-symbol'] = range(1, 11)  # Add ranking to labels
geojsonio.embed(top_ten.to_json())

shpfile = './data/cambridge/TRANS_Centerlines.shp'
roads = gpd.GeoDataFrame.from_file(shpfile)
roads.to_crs(epsg=4326, inplace=True)  # Project local coordinate system into lon/lat

# Find the ID of the closest road to each segment
road_ids = []
for sid, segment in gdf.iterrows():
    distances = roads['geometry'].distance(segment['geometry'])
    road_ids.append(roads['ID'][distances.argmin()])  # Record ID of closest road
    
# Combine with segment data and compute median speeds, segment_counts
gdf['road_id'] = road_ids
grp = gdf.groupby('road_id')
road_speeds = grp['speed'].agg({'median_speed': np.median, 'segment_count': len})

# Merge speeds with road data
roads = roads.merge(road_speeds, left_on='ID', right_index=True)
roads = gpd.GeoDataFrame(roads)  # Ensure type

# Sort intersections by fastest speed
roads.sort(columns=['median_speed'], ascending=False, inplace=True)

# Find fastest stretch of road (of ones with at least 100 segments)
idx = roads['segment_count'] >= 100
fastest_road = roads[idx].head(1)
geojsonio.embed(fastest_road.to_json())

# Export to geoJSON file for importing into Mapbox Studio
f = open('./data/road_speeds.geojson', 'w')
f.write(roads.to_json()) 
f.close()

