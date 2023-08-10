from __future__ import (absolute_import, division, print_function)

from pylab import *

import netCDF4
import pyugrid
import matplotlib.tri as tri

# Read connectivity array using pyugrid

# dataset from FVCOM
# big
url =  'http://testbedapps-dev.sura.org/thredds/dodsC/in/usf/fvcom/ike/ultralite/vardrag/wave/2d'
# little 
url = 'http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM2_FORECAST.nc'

# note: this reads the whole thing in to memory at once: maybe we don't want to do that.
print("Loading data: This could take a while...")
ug = pyugrid.UGrid.from_ncfile(url)

# What's in there?
print("There are %i nodes"%ug.nodes.shape[0])
print("There are %i edges"%ug.edges.shape[0])
print("There are %i faces"%ug.faces.shape[0])


print('The start of the "connectivity array":\n', ug.faces[:5])

# now read the data, because pyugrid doesn't do this yet
nc = netCDF4.Dataset(url)
ncv = nc.variables

nc.variables.keys()

print(ncv['zeta'])

#z = ncv['zeta'][700,:]
z = ncv['zeta'][10,:]

lon = ncv['lon'][:]
lat = ncv['lat'][:]

triang = tri.Triangulation(lon,lat, triangles=ug.faces)

figure(figsize=(12,8))
levs=linspace(floor(z.min()),ceil(z.max()),40)
gca().set_aspect(1./cos(lat.mean()*pi/180))
tricontourf(triang, z,levels=levs)
colorbar()
tricontour(triang, z, colors='k',levels=levs)

