import functools
import geopy
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

print 'This is our latest reading:'
print max(readings['start_datetime'])

print rides.shape
print readings.shape
n, p = readings.shape

readings.ix[:, 0:14].describe()

readings.ix[:, 14:].describe()

rides.describe()

readings.plot(x='duration', y='total_readings', kind='scatter')
plt.title('Verifying that we are sampling at 100 Hz With No Gaps in Data')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

for random_ride_id in np.random.choice(rides.id, 100):
    for i, reading in readings.loc[readings['ride_id'] == random_ride_id, :].iterrows():
        plt.plot([reading['start_x'], reading['end_x']], [reading['start_y'], reading['end_y']])
    plt.title('Plotting Ride ' + str(random_ride_id))
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()

readings['gps_speed'].plot(kind='hist', bins = 100, range=(0, 29))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
print sp.stats.describe(readings['gps_dist'])
print np.percentile(readings['gps_dist'], 5)

readings.plot(x='gps_speed', y='std_z', alpha=0.08, kind='scatter')
fig = plt.gcf()
plt.title('Relationship between Speed and Vibration')
fig.set_size_inches(18.5, 10.5)
plt.xlim(0, 30)
plt.ylim(0, 5)
plt.show()
# ax = sb.regplot(x="total_bill", y="tip", data=tips, scatter_kws={'alpha':0.3})

readings.plot(x='gps_speed', y='abs_sum_z', alpha=0.08, kind='scatter')
fig = plt.gcf()
plt.title('Relationship between Speed and Vibration (Different Measure)')
fig.set_size_inches(18.5, 10.5)
plt.xlim(0, 30)
plt.ylim(0, 500)
plt.show()
# ax = sb.regplot(x="total_bill", y="tip", data=tips, scatter_kws={'alpha':0.3})

for axis in ['x', 'y', 'z']:
    readings['std_' + axis].plot(kind='hist', bins=40)
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    plt.title('Std of ' + axis + ' axis')
    plt.show()

sample_size = 15
indices = np.random.choice(n, sample_size)
for axis in ['x', 'y', 'z']:
    for i in indices:
        sb.tsplot(readings['num_accel_' + axis][i][0:100], alpha=0.50, color=np.random.random(3))
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xlabel('Accelerometer Values')
    plt.ylabel('Force (Gravities)')
    plt.title('Random sample of ' + str(sample_size) + ' ' + axis + ' Accelerometer Time Series')
    plt.show()

sample_size = 1000
indices = np.random.choice(n, sample_size)
for axis in ['x', 'y', 'z']:
    for i in indices:
        f, Pxx_den = signal.periodogram(readings['num_accel_' + axis][i][0:100])
        plt.plot(f, Pxx_den)
        plt.title('Power Spectrum for ' + axis + ' axis')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Power Spectrum Density')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.show()

