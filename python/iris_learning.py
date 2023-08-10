import iris
import datetime
from iris.time import PartialDateTime

import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt

import matplotlib.cm as mpl_cm

import cartopy
import cartopy.crs as ccrs

import numpy

get_ipython().magic('matplotlib inline')

print iris.__version__
print cartopy.__version__

ufile = '/home/dbirving/Downloads/Data/ua_ERAInterim_500hPa_090day-runmean-1989_native.nc'
vfile = '/home/dbirving/Downloads/Data/va_ERAInterim_500hPa_090day-runmean-1989_native.nc'
#/users/damienirving/

# Pick April data
time_constraint = iris.Constraint(time=lambda t: PartialDateTime(month=3) <= t <= PartialDateTime(month=5))
lat_constraint = iris.Constraint(latitude=lambda y: y <= 0.0)

with iris.FUTURE.context(cell_datetime_objects=True):
    u_cube = iris.load_cube(ufile, 'eastward_wind' & time_constraint & lat_constraint)
    v_cube = iris.load_cube(vfile, 'northward_wind' & time_constraint & lat_constraint)

print u_cube

print u_cube.standard_name
print u_cube.long_name
print u_cube.units
print u_cube.name()
print ''
print type(u_cube.data)
print u_cube.data.shape

print 'All times :\n', u_cube.coord('time')

u_temporal_mean = u_cube.collapsed('time', iris.analysis.MEAN)
v_temporal_mean = v_cube.collapsed('time', iris.analysis.MEAN)

print u_temporal_mean

# Load a Cynthia Brewer palette.
brewer_cmap = mpl_cm.get_cmap('brewer_OrRd_09')

# Draw the contours, with n-levels set for the map colours (9).
# NOTE: needed as the map is non-interpolated, but matplotlib does not provide
# any special behaviour for these.
qplt.contourf(u_temporal_mean, brewer_cmap.N, cmap=brewer_cmap)
#qplt.contourf(temporal_mean, 25)
qplt.contour(u_temporal_mean)

# Add coastlines to the map created by contourf.
plt.gca().coastlines()

plt.show()

for coord in u_temporal_mean.coords():
    print coord.name()

test = u_temporal_mean.data
print test.shape

## Define the data
x = u_temporal_mean.coords('longitude')[0].points
y = u_temporal_mean.coords('latitude')[0].points
u = u_temporal_mean.data
v = v_temporal_mean.data

plt.figure(figsize=(8, 10))

## Select the map projection
ax = plt.axes(projection=ccrs.SouthPolarStereo())
#ax = plt.axes(projection=ccrs.Stereographic(central_latitude=0.0, 
#                                            central_longitude=0.0, 
#                                            false_easting=0.0, 
#                                            false_northing=0.0, 
#                                            true_scale_latitude=None, 
#                                            globe=None))
#ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent((x.min(), x.max(), y.min(), -30.0), crs=ccrs.PlateCarree())
## Plot coast and gridlines (currently an error with coastline plotting)
ax.coastlines()
ax.gridlines()
#ax.set_global()

## Plot the data
# Streamplot
magnitude = (u ** 2 + v ** 2) ** 0.5
ax.streamplot(x, y, u, v, transform=ccrs.PlateCarree(), linewidth=2, density=2, color=magnitude)

# Wind vectors
#ax.quiver(x, y, u, v, transform=ccrs.PlateCarree(), regrid_shape=40) 

# Contour
#qplt.contourf(u_temporal_mean)

plt.show()



