#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
import numpy as np
from netCDF4 import Dataset
from matplotlib import animation, rc
from matplotlib import pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

# open a the netCDF file for reading.
filename = r'ctcs_1a.nc'
filename = r'netcdf_2017_10_17/KP07_2017_10_17-14_04_45.nc'
ncfile = Dataset(filename,'r') 

for var in ncfile.variables:
    print var

print ("\nAttributes:")    
for attr in ncfile.ncattrs():
    print attr, "\t --> ", ncfile.getncattr(attr)

eta = ncfile.variables['eta']
t,nx,ny = eta.shape

eta_max = np.max(eta)
eta_min = np.min(eta)

fig = plt.figure()
im = plt.imshow(eta[0,:,:], interpolation='spline36', vmax=eta_max, vmin=eta_min)
plt.axis('equal')
plt.colorbar()

def animate(i):
    im.set_data(eta[i,:,:])        

anim = animation.FuncAnimation(fig, animate, range(t), interval=100)
plt.close(anim._fig)
anim

ncfile.close()

