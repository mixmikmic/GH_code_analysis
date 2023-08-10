import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from quagmire import tools as meshtools
get_ipython().magic('matplotlib inline')

dem = imread('data/port_macquarie.tif', mode='F')

rows, columns = dem.shape
aspect_ratio = float(columns) / float(rows)

spacing = 5.0

minX, maxX = 0.0, spacing*dem.shape[1]
minY, maxY = 0.0, spacing*dem.shape[0]


fig = plt.figure(1, figsize=(10*aspect_ratio,10))
ax = fig.add_subplot(111)
ax.axis('off')
im = ax.imshow(dem, cmap='terrain_r', origin='lower', aspect=aspect_ratio)
fig.colorbar(im, ax=ax, label='height')

gradX, gradY = np.gradient(dem, 5., 5.) # 5m resolution in each direction
slope = np.hypot(gradX, gradY)

print("min/max slope {}".format((slope.min(), slope.max())))

height, width = slope.shape

radius_min = 50.0
radius_max = 100.0

radius = 1.0/(slope + 0.02)
radius = (radius - radius.min()) / (radius.max() - radius.min()) 
radius = radius * (radius_max-radius_min) + radius_min

# apply gaussian filter for better results
from scipy.ndimage import gaussian_filter
radius2 = gaussian_filter(radius, 5.)

# radius -= slope.min()
# radius /= slope.max()/100
# radius += 1e-8

fig = plt.figure(1, figsize=(10*aspect_ratio, 10))
ax = fig.add_subplot(111)
ax.axis('off')
im = ax.imshow((radius2), cmap='jet', origin='lower', aspect=aspect_ratio)
fig.colorbar(im, ax=ax, label='radius2')

plt.show()

x, y, bmask = meshtools.poisson_square_mesh(minX, maxX, minY, maxY, spacing*2.0, boundary_samples=500, r_grid=radius2*2.0)
print("{} samples".format(x.size))

from scipy import ndimage

coords = np.stack((y, x)).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')


fig = plt.figure(1, figsize=(10*aspect_ratio, 10))
ax = fig.add_subplot(111)
ax.axis('off')
sc = ax.scatter(x[bmask], y[bmask], s=1, c=meshheights[bmask])
sc = ax.scatter(x[~bmask], y[~bmask], s=5, c=meshheights[~bmask])

fig.colorbar(sc, ax=ax, label='height')
plt.show()

from quagmire import TopoMesh # all routines we need are within this class
from quagmire import SurfaceProcessMesh

dm = meshtools.create_DMPlex_from_points(x, y, bmask, refinement_steps=2)
mesh = SurfaceProcessMesh(dm)

# Triangulation reorders points
coords = np.stack((mesh.tri.points[:,1], mesh.tri.points[:,0])).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')

mesh.update_height(meshheights)

gradient_max = mesh.slope.max()
gradient_mean = mesh.slope.mean()
flat_spots = np.where(mesh.slope < gradient_mean*0.01)[0]
low_points = mesh.identify_low_points()

nodes = np.arange(0, mesh.npoints)
lows =  np.where(mesh.down_neighbour[1] == nodes)[0]

# print statistics
print("mean gradient {}\nnumber of flat spots {}\nnumber of low points {}".format(gradient_mean,
                                                                                  flat_spots.size,
                                                                                  low_points.shape[0]))

print low_points

flowpaths1 = mesh.cumulative_flow(mesh.area)

filled_height = np.zeros_like(mesh.height)
filled_height[low_points] = ( mesh.height[mesh.neighbour_cloud[low_points,1:7]].mean(axis=1) ) 
raw_heights = mesh.height.copy()


new_heights = mesh.backfill_low_points(50)

fig = plt.figure(1, figsize=(10*aspect_ratio,10))
ax = fig.add_subplot(111)
ax.axis('off')
im1 = ax.tripcolor(mesh.tri.x, mesh.tri.y, mesh.tri.simplices, new_heights-raw_heights, linewidth=0.1, cmap='jet')
fig.colorbar(im1, ax=ax, label='slope')
ax.scatter(mesh.tri.x[low_points], mesh.tri.y[low_points], s=10.0, color="Red")
plt.show()

low_points2 = mesh.identify_low_points()

fig = plt.figure(1, figsize=(10*aspect_ratio,10))
ax = fig.add_subplot(111)
ax.axis('off')
im1 = ax.tripcolor(mesh.tri.x, mesh.tri.y, mesh.tri.simplices, new_heights-raw_heights, linewidth=0.1, cmap='jet')
fig.colorbar(im1, ax=ax, label='slope')
ax.scatter(mesh.tri.x[low_points], mesh.tri.y[low_points], s=1.0, color="Red")
ax.scatter(mesh.tri.x[low_points2], mesh.tri.y[low_points2], s=10.0, color="Green")

plt.show()

its, flowpaths2 = mesh.cumulative_flow_verbose(mesh.area, maximum_its=1000, verbose=True)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    
im1 = ax1.tripcolor(mesh.tri.x, mesh.tri.y, mesh.tri.simplices, np.log(np.sqrt(flowpaths1)) , cmap='Blues')
im2 = ax2.tripcolor(mesh.tri.x, mesh.tri.y, mesh.tri.simplices, np.log(np.sqrt(flowpaths2)) , cmap='Blues')

# fig.colorbar(im1, ax=ax1)
plt.show()







