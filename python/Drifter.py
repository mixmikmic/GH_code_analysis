# Turning on inline plots -- just for use in ipython notebooks.
get_ipython().magic('pylab inline')

import numpy as np
import datetime as dt
import netCDF4 as netCDF
import tracpy
import tracpy.plotting
from tracpy.tracpy_class import Tracpy

# Location of TXLA model output file and grid, on a thredds server.
loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_nesting6.nc'

# Number of days to run the drifters.
ndays = 3

# Start date in date time formatting
date = dt.datetime(2009, 11, 25, 0)

# Time between outputs
tseas = 4*3600 # 4 hours between outputs, in seconds 

# Time units
time_units = 'seconds since 1970-01-01'

# Sets a smaller limit than between model outputs for when to force interpolation if hasn't already occurred.
nsteps = 5
# Controls the sampling frequency of the drifter tracks.
N = 4

# Use ff = 1 for forward in time and ff = -1 for backward in time.
ff = 1

ah = 0. # m^2/s
av = 0. # m^2/s

# turbulence/diffusion flag
doturb = 0

# simulation name, used for saving results into netcdf file
name = 'simulation1'

# for 3d flag, do3d=0 makes the run 2d and do3d=1 makes the run 3d
do3d = 0

## Choose method for vertical placement of drifters
z0 = 's' 
num_layers = 30
zpar = num_layers-1 

# Initialize Tracpy class
tp = Tracpy(loc, name=name, tseas=tseas, ndays=ndays, nsteps=nsteps, usebasemap=True,
            N=N, ff=ff, ah=ah, av=av, doturb=doturb, do3d=do3d, z0=z0, zpar=zpar, time_units=time_units)

# read in grid
tp._readgrid()

# Input starting locations as real space lon,lat locations
lon0, lat0 = np.meshgrid(np.linspace(-98.5,-87.5,55),                             np.linspace(22.5,31,49)) # whole domain, 20 km

# Eliminate points that are outside domain or in masked areas
lon0, lat0 = tracpy.tools.check_points(lon0, lat0, tp.grid)

# Note in timing that the grid was already read in
lonp, latp, zp, t, T0, U, V = tracpy.run.run(tp, date, lon0, lat0)

tracpy.plotting.tracks(lonp, latp, tp.name, tp.grid)

tracpy.plotting.hist(lonp, latp, tp.name, grid=tp.grid, which='hexbin', bins=(50,50))



