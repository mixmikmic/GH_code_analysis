import functools
import geopy
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

reload(data_munging)

rides, readings = data_munging.read_raw_data()
readings = data_munging.clean_readings(readings)
readings = data_munging.add_proj_to_readings(readings, data_munging.NAD83)

n, p = readings.shape
print rides.columns
print readings.columns

kimball_readings = data_munging.filter_readings_by_bb(readings, data_munging.kimball_bounding_box)
chi_readings = data_munging.filter_readings_to_chicago(readings)

groups = kimball_readings.groupby('ride_id')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for ride, group in groups:
    ax.plot(group.start_x, group.start_y, marker='o', linestyle='', ms=12, label=ride)
ax.legend()
fig = plt.gcf()
fig.set_size_inches(30, 30)
plt.show()

chi_readings['std_over_speed'] = chi_readings['std_total'] / chi_readings['gps_speed']
chi_readings['std_over_speed_capped'] = chi_readings['std_over_speed'].apply(lambda x: min(x, 1.0))
for var in ['std_total', 'std_over_speed', 'std_over_speed_capped']:
    plot = chi_readings.plot(x='start_x', y='start_y', kind='scatter', c=var, colormap='Oranges')
    fig = plt.gcf()
    fig.set_size_inches(150, 150)
    plot.set_axis_bgcolor('w')
    plt.show()

kimball_readings['std_over_speed'] = kimball_readings['std_total'] / kimball_readings['gps_speed']
kimball_readings['std_over_speed_capped'] = kimball_readings['std_over_speed'].apply(lambda x: min(x, 1.0))
for var in ['std_total', 'std_over_speed', 'std_over_speed_capped']:
    plot = kimball_readings.plot(x='start_x', y='start_y', kind='scatter', c=var, colormap='Oranges')
    fig = plt.gcf()
    fig.set_size_inches(30, 30)
    plot.set_axis_bgcolor('w')
    plt.show()

kimball_readings['std_over_speed'] = kimball_readings['std_total'] / kimball_readings['gps_speed']
kimball_readings.plot(x='start_x', y='start_y', kind='scatter', c='std_over_speed', colormap='Oranges_r')
fig = plt.gcf()
fig.set_size_inches(30, 30)
plt.show()

readings_idx = data_munging.insert_readings_rtree(kimball_readings)

min(kimball_readings.start_x)

bumpy_idx = [i for i in readings_idx.intersection((351200 + 0, 584800 + 0 , 351200 + 100, 584800 + 650))]
    

kimball_proper_idx = [i for i in readings_idx.intersection((351200 + 280, 584800 + 500 , 351200 + 400, 584800 + 700))]

kimball_readings.loc[kimball_proper_idx, :].plot(x='start_x', y='start_y', kind='scatter', )
fig = plt.gcf()
fig.set_size_inches(30, 30)
plt.show()

kimball_readings.loc[kimball_proper_idx, :].plot(x='gps_speed', y='std_total', kind='scatter')
plt.show()



kimball_readings.loc[bumpy_idx, :].plot(x='start_x', y='start_y', kind='scatter', )
fig = plt.gcf()
fig.set_size_inches(30, 30)
plt.show()

