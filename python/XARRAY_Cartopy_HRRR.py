get_ipython().magic('matplotlib inline')
import xarray
import matplotlib.pyplot as plt
import numpy as np

url = 'http://thredds.ucar.edu/thredds/dodsC/grib/NCEP/HRRR/CONUS_2p5km/Best'

nc = xarray.open_dataset(url)

var='Temperature_height_above_ground'
ncvar = nc[var]
ncvar

grid = nc[ncvar.grid_mapping]
grid

lon0 = grid.longitude_of_central_meridian
lat0 = grid.latitude_of_projection_origin
lat1 = grid.standard_parallel
earth_radius = grid.earth_radius

import cartopy
import cartopy.crs as ccrs

isub = 10

ncvar.x

#cartopy wants meters, not km
x = ncvar.x[::isub].data*1000.
y = ncvar.y[::isub].data*1000.

#globe = ccrs.Globe(ellipse='WGS84') #default
globe = ccrs.Globe(ellipse='sphere', semimajor_axis=grid.earth_radius)

crs = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0, 
                            standard_parallels=(lat0,lat1), globe=globe)

# find the correct time dimension name
for d in ncvar.dims:
    if "time" in d: 
        timevar = d

istep = -1
fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())
mesh = ax.pcolormesh(x,y,ncvar[istep,0,::isub,::isub].data.squeeze()-273.15, transform=crs,zorder=0, vmin=0, vmax=40)
fig.colorbar(mesh)
ax.coastlines(resolution='10m',color='black',zorder=1)
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False
plt.title(nc[timevar].data[istep]);



