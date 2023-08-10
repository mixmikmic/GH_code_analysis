import gpxpy
import mplleaflet
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['axes.xmargin'] = 0.1
plt.rcParams['axes.ymargin'] = 0.1
get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")

with open('../gpx/hh_marathon.gpx') as fh:
    gpx_file = gpxpy.parse(fh)

segment = gpx_file.tracks[0].segments[0]
coords = pd.DataFrame([{'lat': p.latitude, 'lon': p.longitude} for p in segment.points])
plot_coords = coords.ix[::150]

fig = plt.figure()
plt.plot(plot_coords['lon'].values, plot_coords['lat'].values)
plt.plot(plot_coords['lon'].values, plot_coords['lat'].values, 'ro')

mplleaflet.display(fig=fig)

with open('../gpx/3-laender-giro.gpx') as fh:
    gpx_file = gpxpy.parse(fh)

segment = gpx_file.tracks[0].segments[0]
coords = pd.DataFrame([{'lat': p.latitude, 'lon': p.longitude} for p in segment.points])
plot_coords = coords.ix[::150]

fig = plt.figure()
plt.plot(plot_coords['lon'].values, plot_coords['lat'].values)
plt.plot(plot_coords['lon'].values, plot_coords['lat'].values, 'ro')

mplleaflet.display(fig=fig)

