# Let's see if we can read a generic h5 file using netCDF4

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import netCDF4

url='/usgs/data2/rsignell/temp_delta_newer.h5'

nc = netCDF4.Dataset(url)

nc.variables.keys()

lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

g = nc.groups

g

g['2020']['45'].variables.keys()

t = g['2020']['45'].variables['10'][:,:]

print lat.shape,lon.shape,t.shape

plt.pcolormesh(lon,lat,t)
plt.colorbar();



