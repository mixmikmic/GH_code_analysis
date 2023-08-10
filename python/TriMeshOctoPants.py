import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from quagmire import tools as meshtools
get_ipython().magic('matplotlib inline')

from quagmire import FlatMesh 
from quagmire import TopoMesh # all routines we need are within this class
from quagmire import SurfaceProcessMesh

minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,

x1, y1, bmask1 = meshtools.poisson_elliptical_mesh(minX, maxX, minY, maxY, 0.1, 500, r_grid=None)


DM = meshtools.create_DMPlex_from_points(x1, y1, bmask1, refinement_steps=2)
mesh = SurfaceProcessMesh(DM)  ## cloud array etc can surely be done better ... 

x = mesh.coords[:,0]
y = mesh.coords[:,1]
bmask = mesh.bmask

radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)

height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(10.0*theta)**2 ## Less so
height  += 0.5 * (1.0-0.2*radius)

rainfall = np.ones_like(height)
rainfall[np.where( radius > 5.0)] = 0.0 

mesh.downhill_neighbours = 2
mesh.update_height(height)

mo1 = mesh.identify_outflow_points()
i = np.argsort(theta[mo1])
outflows = mo1[i]

mesh.downhill_neighbours = 2
mesh.update_height(height)

flowpaths = mesh.cumulative_flow(rainfall*mesh.area)
logpaths = np.log10(flowpaths)
sqrtpaths = np.sqrt(flowpaths)

mesh.downhill_neighbours = 3
mesh.update_height(height)


flowpaths3 = mesh.cumulative_flow(rainfall*mesh.area)
logpaths3 = np.log10(flowpaths3)
sqrtpaths3 = np.sqrt(flowpaths3)

mesh.downhill_neighbours = 1
mesh.update_height(height)

flowpaths1 = mesh.cumulative_flow(rainfall*mesh.area)
logpaths1 = np.log10(flowpaths1)
sqrtpaths1 = np.sqrt(flowpaths1)

## What's happening with the outflow points - how to find them ? 
"""
circum_points = np.where( np.abs(radius-4.9) <= 0.001 )[0]
circum_angle = theta[circum_points]

circum_flow_1 = flowpaths1[circum_points]
circum_flow_2 = flowpaths[circum_points]
circum_flow_3 = flowpaths3[circum_points]

circum_flow_1n = flowpaths_noise1[circum_points]
circum_flow_2n = flowpaths_noise[circum_points]
circum_flow_3n = flowpaths_noise3[circum_points]
"""
pass

# Choose a scale to plot all six flow results
fmax = 1.0

from scipy import ndimage

fig = plt.figure(1, figsize=(10.0, 10.0))
ax = fig.add_subplot(111)
ax.axis('off')
sc = ax.scatter(x[bmask], y[bmask], s=1, c=mesh.height[bmask], vmin=0.0, vmax=1.0)
sc = ax.scatter(x[~bmask], y[~bmask], s=5, c=mesh.height[~bmask], vmin=0.0, vmax=1.0)

# fig.colorbar(sc, ax=ax, label='height')
plt.show()

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
for ax in [ax1, ax2]:
    ax.axis('equal')
    ax.axis('off')
    
    
im1 = ax1.tripcolor(x, y, mesh.tri.simplices, flowpaths1 ,     cmap='Blues')
im2 = ax2.tripcolor(x, y, mesh.tri.simplices, flowpaths,       cmap="Blues")

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
plt.show()

print mesh.height[mesh.neighbour_cloud[0].compressed()]
print np.argmin(mesh.height[mesh.neighbour_cloud[0].compressed()])

mesh.neighbour_cloud_distances[0]

mesh.near_neighbours_mask.shape

hnear = np.ma.array(mesh.height[mesh.neighbour_cloud], mask=mesh.near_neighbours_mask)
low_neighbours = np.argmin(hnear, axis=1)

hnear = np.ma.array(mesh.height[mesh.neighbour_cloud], mask=mesh.extended_neighbours_mask)
low_eneighbours = np.argmin(hnear, axis=1)


np.where(low_eneighbours == 0)

print mesh.height[mesh.down_neighbour[1][node]] - mesh.height[node], mesh.near_neighbours[node]

deltah = np.ma.array(mesh.height[mesh.neighbour_cloud] - mesh.height.reshape(-1,1), mask = mesh.neighbour_cloud.mask)



ind = np.indices(mesh.neighbour_cloud.shape)[1]
mask = ind > mesh.near_neighbours.reshape(-1,1)

mesh.gaussian_dist_w.shape

near_neighbours = np.ma.array(mesh.neighbour_cloud, mask=mesh.near_neighbours_mask)
print near_neighbours[0]



get_ipython().run_cell_magic('timeit', '', 'for i in range(0, mesh.npoints):\n    np.unique(near_neighbours[near_neighbours[i].compressed()].compressed())')

np.sort(mesh.neighbour_cloud[0].compressed())

import petsc4py

xx = np.linspace(minX, maxX, 250)
yy = np.linspace(minY, maxY, 150)
x1, y1 = np.meshgrid(xx,yy)

x1 += np.random.random(x1.shape) * 0.05 * (maxX-minX) / 250.0
y1 += np.random.random(y1.shape) * 0.05 * (maxY-minY) / 150.0

x1 = x1.flatten()
y1 = y1.flatten()

pts = np.stack((x1, y1)).T

x1.shape



