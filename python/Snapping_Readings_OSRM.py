import collections
import functools
from imposm.parser import OSMParser
import json
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from numpy import nan
import numpy as np
import pandas as pd
import pyproj
import requests
import scipy as sp
import rtree
# import seaborn as sb
from scipy import signal
# import shapely
import shapely.geometry
get_ipython().magic('pylab inline')

import data_munging

rides, readings = data_munging.read_raw_data()
readings = data_munging.clean_readings(readings)
readings = data_munging.add_proj_to_readings(readings, data_munging.NAD83)

digital_ocean_url = 'http://162.243.23.60/osrm-chi-vanilla/'
local_docker_url = 'http://172.17.0.2:5000/'
url = local_docker_url
nearest_request = url + 'nearest?loc={0},{1}'
match_request = url + 'match?loc={0},{1}&t={2}&loc={3},{4}&t={5}'

def readings_to_match_str(readings):
    data_str = '&loc={0},{1}&t={2}'
    output_str = ''
    elapsed_time = 0
    for i, reading in readings.iterrows():
        elapsed_time += 1
        new_str = data_str.format(str(reading['start_lat']), str(reading['start_lon']), str(elapsed_time))
        output_str += new_str
    return url + 'match?' + output_str[1:]

test_request = readings_to_match_str(readings.loc[readings['ride_id'] == 128,  :])
print(test_request)

matched_ride = requests.get(test_request).json()

snapped_points =  pd.DataFrame(matched_ride['matchings'][0]['matched_points'], columns=['lat', 'lon'])

ax = snapped_points.plot(x='lon', y='lat', kind='scatter')
readings.loc[readings['ride_id'] == 128,  :].plot(x='start_lon', y='start_lat', kind='scatter', ax=ax)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

a_reading = readings.loc[0, :]
test_match_request = match_request.format(a_reading['start_lat'],
                                      a_reading['start_lon'], 
                                      0,
                                      a_reading['end_lat'],
                                      a_reading['end_lon'],
                                      1)
# This does not work because OSRM does not accept floats as times. 
# test_map_request = map_request.format(*tuple(a_reading[['start_lat', 'start_lon', 'start_time',
#                                                 'end_lat', 'end_lon', 'end_time']]))

test_nearest_request = nearest_request.format(a_reading['start_lat'], a_reading['start_lon'])

osrm_response = requests.get(test_match_request).json()
osrm_response['matchings'][0]['matched_points']

osrm_response = requests.get(test_nearest_request).json()
osrm_response['mapped_coordinate']

readings['snapped_lat'] = 0
readings['snapped_lon'] = 0

chi_readings = data_munging.filter_readings_to_chicago(readings)
chi_rides = list(set(chi_readings.ride_id))

# This is a small list of rides that I think are bad based upon their graphs.
# I currently do not have an automatic way to update this.
bad_rides = [128, 129, 5.0, 7.0, 131, 133, 34, 169]
good_chi_rides = [i for i in chi_rides if i not in bad_rides]

for ride_id in chi_rides:
    if ride_id in bad_rides:
        print('ride_id')
        try:
            print('num readings: ' + str(sum(readings['ride_id'] == ride_id)))
        except:
            print('we had some issues here.')

all_snapped_points = []
readings['snapped_lat'] = np.NaN
readings['snapped_lon'] = np.NaN
for ride_id in chi_rides:
    if pd.notnull(ride_id):
        ax = readings.loc[readings['ride_id'] == ride_id, :].plot(x='start_lon', y='start_lat')
        try:
            matched_ride = requests.get(readings_to_match_str(readings.loc[readings['ride_id'] == ride_id,  :])).json() 
            readings.loc[readings['ride_id'] == ride_id, ['snapped_lat', 'snapped_lon']] = matched_ride['matchings'][0]['matched_points']
            readings.loc[readings['ride_id'] == ride_id, :].plot(x='snapped_lon', y='snapped_lat', ax=ax)
        except:
            print('could not snap')
        plt.title('Plotting Ride ' + str(ride_id))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()

ax = readings.loc[readings['ride_id'] == 2, :].plot(x='snapped_lon', y='snapped_lat', style='r-')
for ride_id in good_chi_rides:
    print(ride_id)
    try:
#         readings.loc[readings['ride_id'] == ride_id, :].plot(x='start_lon', y='start_lat', ax=ax)
        readings.loc[readings['ride_id'] == ride_id, :].plot(x='snapped_lon', y='snapped_lat', ax=ax, style='b-')
    except:
        print('bad')
ax = readings.loc[readings['ride_id'] == 2, :].plot(x='snapped_lon', y='snapped_lat', style='r-', ax=ax)
fig = plt.gcf()
fig.set_size_inches(36, 36)
plt.show()

# This code goes through a ride backwards in order to figure out what two endpoints 
# the bicycle was going between.
readings['next_snapped_lat'] = np.NaN
readings['next_snapped_lon'] = np.NaN
for ride_id in chi_rides:
    next_lat_lon = (np.NaN, np.NaN)
    for index, row in reversed(list(readings.loc[readings['ride_id'] == ride_id, :].iterrows())):
        readings.loc[index, ['next_snapped_lat', 'next_snapped_lon']] = next_lat_lon
        if (row['snapped_lat'], row['snapped_lon']) != next_lat_lon:
            next_lat_lon = (row['snapped_lat'], row['snapped_lon'])

clean_chi_readings = readings.loc[[ride_id in chi_rides for ride_id in readings['ride_id']], :]

clean_chi_readings.to_csv(data_munging.data_dir + 'clean_chi_readings.csv')

clean_chi_readings = pd.read_csv(data_munging.data_dir + 'clean_chi_readings.csv')

road_bumpiness = collections.defaultdict(list)
for index, reading in clean_chi_readings.iterrows():
    if reading['gps_mph'] < 30 and reading['gps_mph'] > 3:
        osm_segment = [(reading['snapped_lat'], reading['snapped_lon']),
                      (reading['next_snapped_lat'], reading['next_snapped_lon'])]
        osm_segment = sorted(osm_segment)
        if all([lat_lon != (np.NaN, np.NaN) for lat_lon in osm_segment]):
            road_bumpiness[tuple(osm_segment)].append(reading['abs_mean_over_speed'])

# sorted_road_bumpiness = sorted(road_bumpiness.items(), key=lambda i: len(i[1]), reverse=True)

total_road_readings = dict((osm_segment, len(road_bumpiness[osm_segment])) for osm_segment in road_bumpiness)

agg_road_bumpiness = dict((osm_segment, np.mean(road_bumpiness[osm_segment])) for osm_segment in road_bumpiness)

agg_path = data_munging.data_dir + 'agg_road_bumpiness.txt'

with open(agg_path, 'w') as f:
    f.write(str(agg_road_bumpiness))

with open(agg_path, 'r') as f:
    agg_road_bumpiness = f.read()

agg_road_bumpiness = eval(agg_road_bumpiness)

def osm_segment_is_null(osm_segment):
    return (pd.isnull(osm_segment[0][0])
            or pd.isnull(osm_segment[0][1])
            or pd.isnull(osm_segment[1][0])
            or pd.isnull(osm_segment[1][1]))

agg_road_bumpiness = dict((osm_segment, agg_road_bumpiness[osm_segment]) for osm_segment in agg_road_bumpiness if not osm_segment_is_null(osm_segment))

# This is where we filter out all osm segments that are too long

def find_seg_dist(lat_lon):
    return data_munging.calc_dist(lat_lon[0][1], lat_lon[0][0], lat_lon[1][1], lat_lon[1][0])

seg_dist = dict()
for lat_lon in agg_road_bumpiness:
    seg_dist[lat_lon] = data_munging.calc_dist(lat_lon[0][1], lat_lon[0][0], lat_lon[1][1], lat_lon[1][0])

with open('../dat/chi_agg_info.csv', 'w') as f:
    f.write('lat_lon_tuple|agg_road_bumpiness|total_road_readings|seg_dist\n')
    for lat_lon in agg_road_bumpiness:
        if data_munging.calc_dist(lat_lon[0][1], lat_lon[0][0], lat_lon[1][1], lat_lon[1][0]) < 200:
            f.write(str(lat_lon) + '|' + str(agg_road_bumpiness[lat_lon])
                    + '|'  + str(total_road_readings[lat_lon])
                    + '|' + str(seg_dist[lat_lon]) + '\n')

seg_dist[lat_lon]

np.max(agg_road_bumpiness.values())

plt.hist(agg_road_bumpiness.values())

import matplotlib.colors as colors

plasma = cm = plt.get_cmap('plasma')
cNorm  = colors.Normalize(vmin=0, vmax=1.0)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plasma)

for osm_segment, bumpiness in agg_road_bumpiness.items():
#     lat_lon = osm_segment
#     color = (1, 0, 0) if data_munging.calc_dist(lat_lon[0][1], lat_lon[0][0], lat_lon[1][1], lat_lon[1][0]) > 100 else (0, 1, 0)
    plt.plot([osm_segment[0][1], osm_segment[1][1]],
             [osm_segment[0][0], osm_segment[1][0]],
#              color=color)
             color=scalarMap.to_rgba(bumpiness))
fig = plt.gcf()
fig.set_size_inches(24, 48)
plt.show()

filtered_agg_bumpiness = dict((lat_lon, agg_road_bumpiness[lat_lon])
                               for lat_lon in agg_road_bumpiness if find_seg_dist(lat_lon) < 200)

with open(data_dir + 'filtered_chi_road_bumpiness.txt', 'w') as f:
    f.write(str(filtered_agg_bumpiness))

