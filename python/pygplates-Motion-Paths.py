import numpy as np
import pygplates

rotation_filename = 'Data/Seton_etal_ESR2012_2012.1.rot'

# Required parameters for a motion path feature
SeedPoint = (30,78)
MovingPlate = 501
RelativePlate = 301
TimeStep = 2
times = np.arange(0,91,TimeStep)

# Create the motion path feature
digitisation_time = 0
seed_points_at_digitisation_time = pygplates.MultiPointOnSphere([SeedPoint])
motion_path_feature = pygplates.Feature.create_motion_path(
        seed_points_at_digitisation_time,
        times,
        valid_time=(200, 0),
        relative_plate=RelativePlate,
        reconstruction_plate_id = MovingPlate)

rotation_model = pygplates.RotationModel(rotation_filename)

# Create the shape of the motion path
reconstruction_time = 0
reconstructed_motion_paths = []
pygplates.reconstruct(
        motion_path_feature, rotation_model, reconstructed_motion_paths, reconstruction_time,
        reconstruct_type=pygplates.ReconstructType.motion_path)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

get_ipython().magic('matplotlib inline')

# get the reconstructed coordinates into numpy arrays
for reconstructed_motion_path in reconstructed_motion_paths:
    trail = reconstructed_motion_path.get_motion_path().to_lat_lon_array()
    
## Plotting - note that we use the median of the motion path coordinates as the map center
fig = plt.figure(figsize=(10,10), dpi=100)
pmap = Basemap(projection='ortho',lon_0=np.median(trail[:,1]),lat_0=np.median(trail[:,0]),resolution='l')
pmap.drawcoastlines(linewidth=0.25)
pmap.fillcontinents(color='darkkhaki',lake_color='aliceblue')
clip_path = pmap.drawmapboundary(fill_color='aliceblue')
pmap.drawmeridians(np.arange(0,360,15))
pmap.drawparallels(np.arange(-90,90,15))

x, y = pmap(np.flipud(trail[:,1]), np.flipud(trail[:,0]))
pmap.plot(x[0],y[0],'ko')
pmap.plot(x,y,'r')
l1=pmap.scatter(x, y, 60, c=times, marker='h',
                          cmap=plt.cm.jet_r, edgecolor='w', clip_path=clip_path, zorder=2)
plt.title('Motion Path of point on plate %d relative to %d' % (MovingPlate,RelativePlate))

cbar = pmap.colorbar(l1,location='right',pad="5%")
cbar.set_label('Age (Ma)',fontsize=12)
    
plt.show()

# Iterate over each segment in the reconstructed motion path, get the distance travelled by the moving
# plate relative to the fixed plate in each time step
Dist = []
for reconstructed_motion_path in reconstructed_motion_paths:
    for segment in reconstructed_motion_path.get_motion_path().get_segments():
        Dist.append(segment.get_arc_length()*pygplates.Earth.mean_radius_in_kms)

# Get rate of motion as distance per Myr
Rate = np.asarray(Dist)/TimeStep

# Note that the motion path coordinates come out starting with the oldest time and working forwards
# So, to match our 'times' array, we flip the order
Rate = np.flipud(Rate)

fig = plt.figure(figsize=(10,4))
plt.plot(times[:-1],Rate)
plt.xlabel('Time (Ma)')
plt.ylabel('Rate of motion (km/Myr)')
plt.gca().invert_xaxis()
plt.show()

StepRate = np.zeros(len(Rate)*2)
StepRate[::2] = Rate
StepRate[1::2] = Rate

StepTime = np.zeros(len(Rate)*2)
StepTime[::2] = times[:-1]
StepTime[1::2] = times[1:]

fig = plt.figure(figsize=(10,4))
plt.plot(StepTime,StepRate)
plt.xlabel('Time (Ma)')
plt.ylabel('Rate of motion (km/Myr)')
plt.gca().invert_xaxis()
plt.show()

SeedPoint = (10,15)
MovingPlate = 701
RelativePlate = 0
times = np.arange(0,130,10)

# Create a motion path feature
digitisation_time = 0
seed_points_at_digitisation_time = pygplates.MultiPointOnSphere([SeedPoint]) 
motion_path_feature = pygplates.Feature.create_motion_path(
        seed_points_at_digitisation_time,
        times,
        valid_time=(200, 0),
        relative_plate=RelativePlate,
        reconstruction_plate_id = MovingPlate)

# Create the shape of the motion path
reconstruction_time = 0
reconstructed_motion_paths = []
pygplates.reconstruct(
        motion_path_feature, rotation_model, reconstructed_motion_paths, reconstruction_time,
        reconstruct_type=pygplates.ReconstructType.motion_path)

# get the reconstructed coordinates into numpy arrays
for reconstructed_motion_path in reconstructed_motion_paths:
    trail = reconstructed_motion_path.get_motion_path().to_lat_lon_array()
 

plt.plot(times,np.flipud(trail[:,0]))
plt.title('Paleolatitude of point in central Africa')
plt.xlabel('Time (Ma)')
plt.ylabel('Latitude')
plt.gca().grid()
plt.gca().invert_xaxis()
plt.show()

# Define new parameters for the feature that will map the hotspot trail
SeedPoint = (-39,-11)
RelativePlate = 701
times = np.arange(0,131,5)

# Create a motion path feature
digitisation_time = 0
seed_points_at_digitisation_time = pygplates.MultiPointOnSphere([SeedPoint]) # Two lat/lon seed points
motion_path_feature = pygplates.Feature.create_motion_path(
        seed_points_at_digitisation_time,
        times,
        valid_time=(200, 0),
        relative_plate=RelativePlate,
        reconstruction_plate_id=1)

# Create the shape of the motion path
reconstruction_time = 0
reconstructed_motion_paths = []
pygplates.reconstruct(
        motion_path_feature, rotation_model, reconstructed_motion_paths, reconstruction_time,
        reconstruct_type=pygplates.ReconstructType.motion_path)

# get the reconstructed coordinates into numpy arrays
for reconstructed_motion_path in reconstructed_motion_paths:
    trail = reconstructed_motion_path.get_motion_path().to_lat_lon_array()
 

fig = plt.figure(figsize=(8,6),dpi=300)
pmap = Basemap(llcrnrlon=-15,llcrnrlat=-42,urcrnrlon=25,urcrnrlat=-16,            rsphere=(6378137.00,6356752.3142),            resolution='l',projection='merc',            lat_0=0.,lon_0=0.,lat_ts=-10.)
clip_path = pmap.drawmapboundary()
pmap.drawmeridians(np.arange(0, 360, 20), labels=[0,0,0,1],fontsize=10)
pmap.drawparallels(np.arange(-90, 90, 20), labels=[1,0,0,0],fontsize=10)
ax = plt.gca()
pmap.etopo(alpha=0.4)

x,y = pmap(np.flipud(trail[:,1]), np.flipud(trail[:,0]))
pmap.plot(x, y, 'k', clip_path=clip_path, zorder=1)
l1=pmap.scatter(x, y, 60, c=times, marker='h',
                          cmap=plt.cm.jet_r, edgecolor='w', clip_path=clip_path, zorder=2)

cbar = pmap.colorbar(l1,location='right',pad="5%")
cbar.set_label('Time (Ma)',fontsize=12)

plt.show()



