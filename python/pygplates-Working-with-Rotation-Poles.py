import pygplates
import numpy as np

input_rotation_filename = 'Data/Seton_etal_ESR2012_2012.1.rot'

rotation_model = pygplates.RotationModel(input_rotation_filename)


finite_rotation = rotation_model.get_rotation(40.1,801,0,802)
pole_lat,pole_lon,pole_angle = finite_rotation.get_lat_lon_euler_pole_and_angle_degrees()

print 'Finite Pole Lat,Lon,Angle = %f,%f,%f ' % (pole_lat,pole_lon,pole_angle)

finite_rotation = rotation_model.get_rotation(102,101,0,201)
pole_lat,pole_lon,pole_angle = finite_rotation.get_lat_lon_euler_pole_and_angle_degrees()

print 'Finite Pole Lat,Lon,Angle = %f,%f,%f ' % (pole_lat,pole_lon,pole_angle)

fixed_plate = 0
moving_plate = 802
to_time = 50
from_time = 55

stage_rotation = rotation_model.get_rotation(to_time,moving_plate,from_time,fixed_plate)

print stage_rotation

pole_lat,pole_lon,pole_angle = stage_rotation.get_lat_lon_euler_pole_and_angle_degrees()

print 'Stage Pole Lat,Lon,Angle = %f,%f,%f ' % (pole_lat,pole_lon,pole_angle)

time_step = 10
for time in np.arange(0,100,time_step):

    to_time = time
    from_time = time+time_step
    stage_rotation = rotation_model.get_rotation(to_time,moving_plate,from_time,fixed_plate)

    pole_lat,pole_lon,pole_angle = stage_rotation.get_lat_lon_euler_pole_and_angle_degrees()

    print 'Time interval = ',time,'-',time+time_step,', Stage Pole Lat,Lon,Angle = %f,%f,%f ' % (pole_lat,pole_lon,pole_angle)
        

fixed_plate = 901
moving_plate = 804
time_step = 1

Lats = []
Longs = []
Angles = []

for time in np.arange(0,42,time_step):
    
    to_time = time
    from_time = time+time_step
    stage_rotation = rotation_model.get_rotation(to_time,moving_plate,from_time,fixed_plate)

    pole_lat,pole_lon,pole_angle = stage_rotation.get_lat_lon_euler_pole_and_angle_degrees()

    Lats.append(pole_lat)
    Longs.append(pole_lon)
    Angles.append(np.radians(pole_angle))
    
# These next lines are necessary becuase the answers come out in the northern hemisphere, 
# need to check convention
Longs = np.add(Longs,180.)
Lats = np.multiply(Lats,-1)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

get_ipython().magic('matplotlib inline')

# Make the figure and a dummy orthographic map to get extents
fig = plt.figure(figsize=(12,6),dpi=300)
lat_0=-75. ; lon_0=130.
m1 = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution=None)

# First subplot
ax = fig.add_axes([0.0,0.1,0.5,0.8],axisbg='k')
pmap = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution='l',    llcrnrx=m1.urcrnrx/-6.,llcrnry=m1.urcrnry/-6.,urcrnrx=m1.urcrnrx/6.,urcrnry=m1.urcrnry/6.)
clip_path = pmap.drawmapboundary(fill_color='white')
pmap.fillcontinents(color='grey', lake_color='white', zorder=0)
pmap.drawmeridians(np.arange(0, 360, 30))
pmap.drawparallels(np.arange(-90, 90, 30))
ax = plt.gca()

x,y = pmap(Longs,Lats)
pmap.plot(x, y, 'r', clip_path=clip_path,zorder=0)
l3=pmap.scatter(x, y, 200, c=np.arange(0,42,time_step),
            cmap=plt.cm.jet_r,edgecolor='k',clip_path=clip_path,vmin=0,vmax=45)

cbar = pmap.colorbar(l3,location='right',pad="5%")
cbar.set_label('Time (Ma)',fontsize=12)

# Second subplot
ax = fig.add_axes([0.5,0.1,0.5,0.8],axisbg='k')
pmap = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution='l',    llcrnrx=m1.urcrnrx/-6.,llcrnry=m1.urcrnry/-6.,urcrnrx=m1.urcrnrx/6.,urcrnry=m1.urcrnry/6.)
clip_path = pmap.drawmapboundary(fill_color='white')
pmap.fillcontinents(color='grey', lake_color='white', zorder=0)
pmap.drawmeridians(np.arange(0, 360, 30))
pmap.drawparallels(np.arange(-90, 90, 30))
ax = plt.gca()

x,y = pmap(Longs,Lats)
pmap.plot(x, y, 'r', clip_path=clip_path,zorder=0)
l3=pmap.scatter(x, y, 200, c=np.degrees(Angles),
            cmap=plt.cm.jet_r,edgecolor='k',clip_path=clip_path,vmin=0.5,vmax=1.2)

cbar = pmap.colorbar(l3,location='right',pad="5%")
cbar.set_label('Angular Rate (deg/Myr)',fontsize=12)

plt.show()



