import gmt

fig = gmt.Figure()

fig.coast(region=[-90, -70, 0, 20], projection='M6i', land='chocolate', 
          frame=True)

fig.show()

fig_alias = gmt.Figure()
fig_alias.coast(R='-90/-70/0/20', J='M6i', G='gray', S="blue", B=True)
fig_alias.show()

fig.savefig('first-steps-central-america.png')

import numpy as np

# See the random number generator so we always 
# get the same numbers
np.random.seed(42)
ndata = 100
region = [150, 240, -10, 60]
# Create some fake distribution of points and a measured value
x = np.random.uniform(region[0], region[1], ndata)
y = np.random.uniform(region[2], region[3], ndata)
magnitude = np.random.uniform(1, 5, size=ndata)

fig = gmt.Figure()
# Create a 6x6 inch basemap using the data region
fig.basemap(region=region, projection='X6i', frame=True)
# Plot using triangles (i) of 0.3 cm
fig.plot(x, y, style='i0.3c', color='black')
fig.show()

fig = gmt.Figure()
fig.basemap(region=region, projection='X6i', frame=True)
fig.plot(x, y, style='ic', color='black', sizes=magnitude/10)
fig.show()

# Save our fake data to a file.
np.savetxt('first-steps-data.txt', 
           np.transpose([x, y, magnitude]))

fig = gmt.Figure()
fig.basemap(region=region, projection='X6i', frame=True)
fig.plot(data='first-steps-data.txt', style='cc', color='red', 
         columns=[0, 1, '2s0.1'])
fig.show()

from gmt.datasets import load_japan_quakes

quakes = load_japan_quakes()
quakes.head()

quakes_region = [quakes.longitude.min() - 1, quakes.longitude.max() + 1,
                 quakes.latitude.min() - 1, quakes.latitude.max() + 1]

fig = gmt.Figure()
fig.coast(region=quakes_region, projection='M6i', frame=True, 
          land='black', water='skyblue')
fig.plot(x=quakes.longitude, y=quakes.latitude, 
         style='c0.3c', color='white', pen='black')
fig.show()

fig = gmt.Figure()
fig.coast(region=quakes_region, projection='M6i', frame=True, 
          land='black', water='skyblue')
fig.plot(x=quakes.longitude, y=quakes.latitude, 
         sizes=0.02*(2**quakes.magnitude),
         style='cc', color='white', pen='black')
fig.show()

fig = gmt.Figure()
fig.coast(region=quakes_region, projection='M6i', frame=True, 
          land='black', water='skyblue')
fig.plot(x=quakes.longitude, y=quakes.latitude, 
         sizes=0.02*2**quakes.magnitude,
         color=quakes.depth_km/quakes.depth_km.max(),
         cmap='viridis', style='cc', pen='black')
fig.show()

fig = gmt.Figure()
fig.coast(region=quakes_region, projection='X6id/6id', land='gray')
fig.plot(x=quakes.longitude, y=quakes.latitude,
         sizes=0.02*2**quakes.magnitude,
         color=quakes.depth_km/quakes.depth_km.max(),
         cmap='viridis', style='cc', pen='black')
fig.show(method='globe')

