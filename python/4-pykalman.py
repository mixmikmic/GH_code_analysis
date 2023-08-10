import gpxpy
import mplleaflet
import pandas as pd
import numpy as np
from pykalman import KalmanFilter

import matplotlib.pyplot as plt
#plt.rcParams['axes.xmargin'] = 0.1
#plt.rcParams['axes.ymargin'] = 0.1
get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")

from gps_utils import haversine

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

with open('../gpx/3-laender-giro.gpx') as fh:
    gpx_file = gpxpy.parse(fh)
    
segment = gpx_file.tracks[0].segments[0]

segment.get_uphill_downhill()

segment.points[0].speed = 0.0
segment.points[-1].speed = 0.0
gpx_file.add_missing_speeds()

speed = np.array([p.speed for p in segment.points])*3.6
plt.plot(speed)

coords = pd.DataFrame([{'idx': i,
                        'lat': p.latitude, 
                        'lon': p.longitude, 
                        'ele': p.elevation,
                        'speed': p.speed,
                        'time': p.time} for i, p in enumerate(segment.points)])
coords.set_index('time', inplace=True)
coords.head()

coords.tail()

coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')
coords.tail()

plt.plot(np.diff(coords.index))

coords = coords.resample('1S').asfreq()
coords.loc[coords.ele.isnull()].head()

plt.plot(np.diff(coords.index))

measurements = np.ma.masked_invalid(coords[['lon', 'lat', 'ele']].values)

fig = plt.figure()
plt.plot(measurements[:,0], measurements[:,1])
filled_coords = coords.fillna(method='pad').ix[coords.ele.isnull()]
plt.plot(filled_coords['lon'].values, filled_coords['lat'].values, 'ro')

F = np.array([[1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

R = np.diag([1e-4, 1e-4, 100])**2

initial_state_mean = np.hstack([measurements[0, :], 3*[0.]])
# works initial_state_covariance = np.diag([1e-3, 1e-3, 100, 1e-4, 1e-4, 1e-4])**2
initial_state_covariance = np.diag([1e-4, 1e-4, 50, 1e-6, 1e-6, 1e-6])**2

kf = KalmanFilter(transition_matrices=F, 
                  observation_matrices=H, 
                  observation_covariance=R,
                  initial_state_mean=initial_state_mean,
                  initial_state_covariance=initial_state_covariance,
                  em_vars=['transition_covariance'])

# Careful here, expectation maximation takes several hours!
#kf = kf.em(measurements, n_iter=1000)
# or just run this instead of the one above (it is the same result)
Q = np.array([[  3.17720723e-09,  -1.56389148e-09,  -2.41793770e-07,
                 2.29258935e-09,  -3.17260647e-09,  -2.89201471e-07],
              [  1.56687815e-09,   3.16555076e-09,   1.19734906e-07,
                 3.17314157e-09,   2.27469595e-09,  -2.11189940e-08],
              [ -5.13624053e-08,   2.60171362e-07,   4.62632068e-01,
                 1.00082746e-07,   2.81568920e-07,   6.99461902e-05],
              [  2.98805710e-09,  -8.62315114e-10,  -1.90678253e-07,
                 5.58468140e-09,  -5.46272629e-09,  -5.75557899e-07],
              [  8.66285671e-10,   2.97046913e-09,   1.54584155e-07,
                 5.46269262e-09,   5.55161528e-09,   5.67122163e-08],
              [ -9.24540217e-08,   2.09822077e-07,   7.65126136e-05,
                 4.58344911e-08,   5.74790902e-07,   3.89895992e-04]])
Q = 0.5*(Q + Q.T) # assure symmetry
kf.transition_covariance = Q

state_means, state_vars = kf.smooth(measurements)

plt.plot(measurements[:,2])

plt.plot(state_means[:,2])

coords.ix[:, ['lon', 'lat', 'ele']] = state_means[:,:3]
orig_coords = coords.ix[~coords['idx'].isnull()].set_index('idx')

for i, p in enumerate(segment.points):
    p.speed = None
    p.elevation = orig_coords.at[float(i),'ele']
    p.longitude = orig_coords.at[float(i),'lon']
    p.latitude = orig_coords.at[float(i),'lat']

segment.get_uphill_downhill()

segment.points[0].speed = 0.0
segment.points[-1].speed = 0.0
gpx_file.add_missing_speeds()

speed = np.array([p.speed for p in segment.points])*3.6
plt.plot(speed)

np.argsort(speed)[:-10:-1]

plt.plot(measurements[1540:1580,0], measurements[1540:1580,1], 'o', alpha=0.5)

segment.points[1545:1558] # speed is in m/s

bad_readings = np.argsort(np.trace(state_vars[:,:2,:2], axis1=1, axis2=2))[:-20:-1]
bad_readings = np.array([idx for idx in range(measurements.shape[0]) if np.min(np.abs(bad_readings - idx)) <= 5])
measurements.mask[bad_readings, :] = True 

state_means, state_vars = kf.smooth(measurements)

coords.ix[:, ['lon', 'lat', 'ele']] = state_means[:,:3]
orig_coords = coords.ix[~coords['idx'].isnull()].set_index('idx')
for i, p in enumerate(segment.points):
    p.speed = None
    p.elevation = orig_coords.at[float(i),'ele']
    p.longitude = orig_coords.at[float(i),'lon']
    p.latitude = orig_coords.at[float(i),'lat']

segment.points[0].speed = 0.0
segment.points[-1].speed = 0.0
gpx_file.add_missing_speeds()

speed = np.array([p.speed for p in segment.points])*3.6
plt.plot(speed)

# calculate the speed directly on our array
speed = [3.6*haversine(state_means[i,1::-1], state_means[i+1,1::-1]) for i in np.arange(state_means.shape[0]-1)]
np.argsort(speed)[:-10:-1]

plt.plot(speed[2450:2650], '.')

plt.plot(measurements[2450:2650,0], measurements[2450:2650,1], '.')

# we check for strong accelerations/deaccelarations
acc = np.gradient(speed)
outliers = np.argsort(np.abs(acc))[:-40:-1]
outliers

outliers = np.array([idx for idx in range(measurements.shape[0]) if np.min(np.abs(outliers - idx)) <= 12])
measurements.mask[outliers] = True
state_means, state_vars = kf.smooth(measurements)

# we smooth several times
for _ in range(10):
    state_means, state_vars = kf.smooth(state_means[:,:3])

speed = [3.6*haversine(state_means[i,1::-1], state_means[i+1,1::-1]) for i in np.arange(state_means.shape[0]-1)]
plt.plot(speed)

plt.plot(speed[2450:2650], '.')

plt.plot(state_means[2450:2650,0], state_means[2450:2650,1], 'r.')
plt.plot(measurements[2450:2650,0], measurements[2450:2650,1], '.')

coords.ix[:, ['lon', 'lat', 'ele']] = state_means[:,:3]
orig_coords = coords.ix[~coords['idx'].isnull()].set_index('idx')

for i, p in enumerate(segment.points):
    p.speed = None
    p.elevation = orig_coords.at[float(i),'ele']
    p.longitude = orig_coords.at[float(i),'lon']
    p.latitude = orig_coords.at[float(i),'lat']

segment.get_uphill_downhill()

coords.ix[:, ['lon', 'lat', 'ele']] = state_means[:,:3]
orig_coords = coords.ix[~coords['idx'].isnull()].set_index('idx')
for i, p in enumerate(segment.points):
    p.speed = None
    p.elevation = orig_coords.at[float(i),'ele']
    p.longitude = orig_coords.at[float(i),'lon']
    p.latitude = orig_coords.at[float(i),'lat']
segment.points[0].speed = 0.0
segment.points[-1].speed = 0.0
gpx_file.add_missing_speeds()
speed = np.array([p.speed for p in segment.points])*3.6
plt.plot(speed)

with open('../gpx/3-laender-giro_cleaned.gpx', 'w') as fh:
    fh.write(gpx_file.to_xml())

