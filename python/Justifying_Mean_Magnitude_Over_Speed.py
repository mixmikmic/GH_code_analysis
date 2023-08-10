import functools
import geopy
import itertools
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import requests
import scipy as sp
import rtree
import seaborn as sb
from scipy import signal
# import shapely
import shapely.geometry
get_ipython().magic('pylab inline')

import data_munging

rides, readings = data_munging.read_raw_data()
readings = data_munging.clean_readings(readings)
readings = data_munging.add_proj_to_readings(readings, data_munging.NAD83)

n, p = readings.shape

readings_idx = data_munging.insert_readings_rtree(readings)

total_intersections = []
for sample_in in range(100):
    random_idx, random_point = data_munging.select_random_point(readings)
    random_bb = data_munging.point_to_bb(*random_point, side_length=0.2)
    intersecting_segments = list(readings_idx.intersection(random_bb))
    intersecting_segments = [i for i in intersecting_segments if data_munging.calc_reading_diffs(readings.ix[i, :], readings.ix[random_idx, :]) < 0.5]
    if len(intersecting_segments) > 1:
        for i in intersecting_segments:
            plt.plot([readings.ix[i, 'start_x'], readings.ix[i, 'end_x']],
                     [readings.ix[i, 'start_y'], readings.ix[i, 'end_y']],
                    alpha=0.5)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.title('Intersections of ' + str(intersecting_segments))
        plt.show()

total_data_points = 0
total_tries = 0
examined_ids = set()
ids0 = list()
ids1 = list()
while total_data_points < 5000 and total_tries < 100000:
    total_tries += 1
    random_idx, random_point = data_munging.select_random_point(readings)
    random_bb = data_munging.point_to_bb(*random_point, side_length=0.2)
    intersecting_segments = list(readings_idx.intersection(random_bb))
    intersecting_segments = set([i for i in intersecting_segments if data_munging.calc_reading_diffs(readings.ix[i, :], readings.ix[random_idx, :]) < 0.4])
    intersecting_segments.difference_update(examined_ids)
    examined_ids.update(intersecting_segments)
    if intersecting_segments > 1:
        for idx_pair in itertools.combinations(intersecting_segments, 2):
            ids0.append(idx_pair[0])
            ids1.append(idx_pair[1])
            total_data_points += 1

pairwise_comps = pd.DataFrame({'id0': ids0, 'id1': ids1})

print len(readings.index)
print len(set(readings.index))

print len(pairwise_comps.index)
print len(set(pairwise_comps.index))

for i in ('0', '1'):
    for var in ('std_z', 'gps_speed', 'abs_mean_total'):
        pairwise_comps[var + i] = readings[var][pairwise_comps['id' + i]].values
for i in ('0', '1'):
    pairwise_comps['abs_mean_over_speed' + i] = pairwise_comps['abs_mean_total' + i] /  pairwise_comps['gps_speed' + i]

pairwise_comps.plot(x='abs_mean_total0', y='abs_mean_total1', kind='scatter', alpha=0.5)
fig = plt.gcf()
fig.set_size_inches(18.5, 18.5)
plt.title('Comparing Total Bumpiness ' + str(intersecting_segments))
plt.show()

pairwise_comps.plot(x='abs_mean_over_speed0', y='abs_mean_over_speed1', kind='scatter', alpha=0.5)
fig = plt.gcf()
fig.set_size_inches(18.5, 18.5)
plt.title('Comparing Total Bumpiness ' + str(intersecting_segments))
plt.show()

pairwise_comps.plot(x='abs_mean_over_speed0', y='abs_mean_over_speed1', kind='scatter', alpha=0.5)
fig = plt.gcf()
fig.set_size_inches(18.5, 18.5)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title('Comparing Total Bumpiness ' + str(intersecting_segments))
plt.show()

np.corrcoef(pairwise_comps['abs_mean_total0'], pairwise_comps['abs_mean_total1'])

np.corrcoef(pairwise_comps['abs_mean_over_speed0'], pairwise_comps['abs_mean_over_speed1'])

