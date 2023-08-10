import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
get_ipython().magic('matplotlib inline')
#import plot_lidar
from datetime import datetime

swath = np.genfromtxt('../../PhD/python-phd/swaths/is6_f11_pass1_aa_nr2_522816_523019_c.xyz')

import pandas as pd

columns = ['time', 'X', 'Y', 'Z', 'I','A', 'x_u', 'y_u', 'z_u', '3D_u']
swath = pd.DataFrame(swath, columns=columns)

swath[1:5]

air_traj = np.genfromtxt('../../PhD/is6_f11/trajectory/is6_f11_pass1_local_ice_rot.3dp')

columns = ['time', 'X', 'Y', 'Z', 'R', 'P', 'H', 'x_u', 'y_u', 'z_u', 'r_u', 'p_u', 'h_u']
air_traj = pd.DataFrame(air_traj, columns=columns)

air_traj[1:5]

fig = plt.figure(figsize = ([30/2.54, 6/2.54]))
ax0 = fig.add_subplot(111)      
a0 = ax0.scatter(swath['Y'],  swath['X'], c=swath['Z'] - np.min(swath['Z']), cmap = 'gist_earth',
                vmin=0, vmax=10, edgecolors=None,lw=0, s=0.6)
a1 = ax0.scatter(air_traj['Y'], air_traj['X'], c=air_traj['Z'], cmap = 'Reds',
                lw=0, s=1)
plt.tight_layout()

import h5py

#create a file instance, with the intention to write it out
lidar_test = h5py.File('lidar_test.hdf5', 'w')

swath_data = lidar_test.create_group('swath_data')

swath_data.create_dataset('GPS_SOW', data=swath['time'])

#some data
swath_data.create_dataset('UTM_X', data=swath['X'])
swath_data.create_dataset('UTM_Y', data=swath['Y'])
swath_data.create_dataset('Z', data=swath['Z'])
swath_data.create_dataset('INTENS', data=swath['I'])
swath_data.create_dataset('ANGLE', data=swath['A'])
swath_data.create_dataset('X_UNCERT', data=swath['x_u'])
swath_data.create_dataset('Y_UNCERT', data=swath['y_u'])
swath_data.create_dataset('Z_UNCERT', data=swath['z_u'])
swath_data.create_dataset('3D_UNCERT', data=swath['3D_u'])

#some attributes
lidar_test.attrs['file_name'] = 'lidar_test.hdf5'

lidar_test.attrs['codebase'] = 'https://github.com/adamsteer/matlab_LIDAR'

traj_data = lidar_test.create_group('traj_data')

#some attributes
traj_data.attrs['flight'] = 11
traj_data.attrs['pass'] = 1
traj_data.attrs['source'] = 'RAPPLS flight 11, SIPEX-II 2012'

#some data
traj_data.create_dataset('pos_x', data = air_traj['X'])
traj_data.create_dataset('pos_y', data =  air_traj['Y'])
traj_data.create_dataset('pos_z', data =  air_traj['Z'])

lidar_test.close()

photo = np.genfromtxt('/Users/adam/Documents/PhD/is6_f11/photoscan/is6_f11_photoscan_Cloud.txt',skip_header=1)

columns = ['X', 'Y', 'Z', 'R', 'G', 'B']
photo = pd.DataFrame(photo[:,0:6], columns=columns)

#create a file instance, with the intention to write it out
lidar_test = h5py.File('lidar_test.hdf5', 'r+')

photo_data = lidar_test.create_group('3d_photo')

photo_data.create_dataset('UTM_X', data=photo['X'])
photo_data.create_dataset('UTM_Y', data=photo['Y'])
photo_data.create_dataset('Z', data=photo['Z'])
photo_data.create_dataset('R', data=photo['R'])
photo_data.create_dataset('G', data=photo['G'])
photo_data.create_dataset('B', data=photo['B'])

#del lidar_test['3d_photo']

lidar_test.close()

from netCDF4 import Dataset

thedata = Dataset('lidar_test.hdf5', 'r')

thedata

swath = thedata['swath_data']

swath

utm_xy = np.column_stack((swath['UTM_X'],swath['UTM_Y']))

idx = np.where((utm_xy[:,0] > -100) & (utm_xy[:,0] < 200) & (utm_xy[:,1] > -100) & (utm_xy[:,1] < 200)  )

chunk_z = swath['Z'][idx]
chunk_z.size

max(chunk_z)

chunk_x = swath['UTM_X'][idx]
chunk_x.size

chunk_y = swath['UTM_Y'][idx]
chunk_y.size

chunk_uncert = swath['Z_UNCERT'][idx]
chunk_uncert.size

plt.scatter(chunk_x, chunk_y, c=chunk_z, lw=0, s=2)

traj = thedata['traj_data']

traj

pos_y = traj['pos_y']

idx = np.where((pos_y[:] > -100.) & (pos_y[:] < 200.))

cpos_x = traj['pos_x'][idx]

cpos_y = traj['pos_y'][idx]

cpos_z = traj['pos_z'][idx]

plt.scatter(chunk_x, chunk_y, c=chunk_z, lw=0, s=3, cmap='gist_earth')
plt.scatter(cpos_x, cpos_y, c=cpos_z, lw=0, s=5, cmap='Oranges')

from mpl_toolkits.mplot3d import Axes3D

#set up a plot
plt_az=310
plt_elev = 40.
plt_s = 3
cb_fmt = '%.1f'

cmap1 = plt.get_cmap('gist_earth', 10)

#make a plot
fig = plt.figure()
fig.set_size_inches(35/2.51, 20/2.51)
ax0 = fig.add_subplot(111, projection='3d')
a0 = ax0.scatter(chunk_x, chunk_y, (chunk_z-min(chunk_z))*2,
                 c=np.ndarray.tolist((chunk_z-min(chunk_z))*2),\
                cmap=cmap1,lw=0, vmin = -0.5, vmax = 5, s=plt_s)
ax0.scatter(cpos_x, cpos_y, cpos_z, c=np.ndarray.tolist(cpos_z),                cmap='hot', lw=0, vmin = 250, vmax = 265, s=10)
ax0.view_init(elev=plt_elev, azim=plt_az)
plt.tight_layout()

#set up a plot
plt_az=310
plt_elev = 40.
plt_s = 3
cb_fmt = '%.1f'

cmap1 = plt.get_cmap('gist_earth', 30)

#make a plot
fig = plt.figure()
fig.set_size_inches(35/2.51, 20/2.51)
ax0 = fig.add_subplot(111, projection='3d')
a0 = ax0.scatter(chunk_x, chunk_y, (chunk_z-min(chunk_z))*2,
                 c=np.ndarray.tolist(chunk_uncert),\
                 cmap=cmap1, lw=0, vmin = 0, vmax = 0.2, s=plt_s)
ax0.scatter(cpos_x, cpos_y, cpos_z, c=np.ndarray.tolist(cpos_z),                cmap='hot', lw=0, vmin = 250, vmax = 265, s=10)
ax0.view_init(elev=plt_elev, azim=plt_az)
plt.tight_layout()
plt.savefig('thefig.png')

photo = thedata['3d_photo']

photo

photo_xy = np.column_stack((photo['UTM_X'],photo['UTM_Y']))

idx_p = np.where((photo_xy[:,0] > 0) & (photo_xy[:,0] < 100) & (photo_xy[:,1] > 0) & (photo_xy[:,1] < 100)  )

plt.scatter(photo['UTM_X'][idx_p], photo['UTM_Y'][idx_p], c = photo['Z'][idx_p],                cmap='hot',vmin=-1, vmax=1, lw=0, s=plt_s)

p_x = photo['UTM_X'][idx_p]
p_y = photo['UTM_Y'][idx_p]
p_z = photo['Z'][idx_p]

plt_az=310
plt_elev = 70.
plt_s = 2

#make a plot
fig = plt.figure()
fig.set_size_inches(25/2.51, 10/2.51)
ax0 = fig.add_subplot(111, projection='3d')

#LiDAR points
ax0.scatter(chunk_x, chunk_y, chunk_z-50,             c=np.ndarray.tolist(chunk_z),            cmap=cmap1, vmin=-30, vmax=2, lw=0, s=plt_s)

#3D photogrammetry pointd
ax0.scatter(p_x, p_y, p_z, 
            c=np.ndarray.tolist(p_z),\
            cmap='hot', vmin=-1, vmax=1, lw=0, s=5)

#aicraft trajectory
ax0.scatter(cpos_x, cpos_y, cpos_z, c=np.ndarray.tolist(cpos_z),                cmap='hot', lw=0, vmin = 250, vmax = 265, s=10)


ax0.view_init(elev=plt_elev, azim=plt_az)
plt.tight_layout()
plt.savefig('with_photo.png')

print('LiDAR points: {0}\nphotogrammetry points: {1}\ntrajectory points: {2}'.
      format(len(chunk_x), len(p_x), len(cpos_x) ))



