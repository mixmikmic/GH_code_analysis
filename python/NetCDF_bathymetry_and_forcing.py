#Lets have matplotlib "inline"
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

#Import packages we need
from netCDF4 import Dataset as NetCDFFile
import numpy as np
from matplotlib import rc
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import mpl_toolkits.basemap.pyproj as pyproj

import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))

#Set large figure sizes
rc('figure', figsize=(16.0, 12.0))
rc('animation', html='html5')

#Import our simulator
from SWESimulators import FBL, CTCS, KP07, CDKLM16, PlotHelper, Common, WindStress
#Import initial condition and bathymetry generating functions:
from SWESimulators.BathymetryAndICs import *

# Read in netCDF file

#url_meps = 'http://thredds.met.no/thredds/dodsC/meps25files/meps_det_pp_2_5km_latest.nc' # using pp-file is faster for testing
url_meps = 'http://thredds.met.no/thredds/dodsC/meps25files/meps_det_extracted_2_5km_latest.nc' 
url_roms = 'http://thredds.met.no/thredds/dodsC/fou-hi/nordic4km-1h/Nordic-4km_SURF_1h_avg_00.nc'

meps_nc = NetCDFFile(url_meps)
roms_nc = NetCDFFile(url_roms)

#print meps_nc
#print roms_nc

# Proj4 strings
meps_proj4_string = meps_nc.variables['projection_lambert'].proj4
roms_proj4_string = roms_nc.variables['polar_stereographic'].proj4

print(meps_proj4_string)
print(roms_proj4_string)

# Variables
lon = meps_nc.variables['longitude'][:]
lat = meps_nc.variables['latitude'][:]
x_wind = meps_nc.variables['x_wind_10m'][:]
y_wind = meps_nc.variables['y_wind_10m'][:]
#x_wind = meps_nc.variables['x_wind_pl'][:] # why can I read every other *wind* variable in the file, except {x,y}_wind_pl???!!!
#y_wind = meps_nc.variables['y_wind_pl'][:]
wind_speed = meps_nc.variables['wind_speed'][:]
h = roms_nc.variables['h'][:]

print(lon[:].shape)
print(lat[:].shape)
print(x_wind[0][:].shape)
print(y_wind[0][:].shape)
print(wind_speed[0][0][:].shape)
print(h[:].shape)

#plt.quiver(x_wind[0][0], y_wind[0][0], wind_speed[0][0])
#plt.show()

fig, ax = plt.subplots()
ax.imshow(x_wind[0][0])
#plt.show()
#ax.contour(lon, colors='black')
#plt.show()
#ax.contour(lat, colors='black')
#plt.show()

# Arome data is in 'lcc' projection
# ROMS data is in 'npstere' projection

m = Basemap(projection='cea', llcrnrlat=-90, urcrnrlat=90, 
llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines()
plt.show()

# CONT HERE: Proj->Basemap??? Remaining tasks in https://github.com/metno/gpu-ocean/pull/84
# Basemap from proj4
meps_m = pyproj.Proj(str(meps_proj4_string))
meps_m.drawcoastlines()
plt.show()
roms_m = pyproj.Proj(str(roms_proj4_string))
roms_m.drawcoastlines()
plt.show()



