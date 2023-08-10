from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma

filename = "tos_O1_2001-2002.nc"
ds = Dataset(filename, mode="r")

print(ds)

md = ds.__dict__
for k in md:
    print("{0}: {1}".format(k, md[k]))

print(ds.dimensions)

print(ds.variables['tos'])

time = ds.variables['time']
print(time)
print(time[:])
lats = ds.variables['lat'][:]
lons = ds.variables['lon'][:]
tos = ds.variables["tos"][:,:,:]
print(tos[0,:,:])

print ('time from {0} to {1}'.format(time[0], time[-1]))
tos_min = ma.min(tos)
tos_max = ma.max(tos)
tos_avg = ma.average(tos)
tos_med = ma.median(tos)[0]
print("Sea surface temperature: min={0}, max={1}".format(tos_min, tos_max))
print("Sea surface temperature: average={0}, median={1}".format(tos_avg, tos_med))

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
cp = plt.contour(tos[12,:,:], 100)

cp = plt.contour(tos[12,:,:], 100)
cbar = plt.colorbar(cp)
cbar.set_label("Temperature [K]")
plt.title("Sea surface temperature")

from mpl_toolkits.basemap import Basemap

fig=plt.figure(figsize=(16,16))

# Create the map
m = Basemap(llcrnrlon=np.min(lons),llcrnrlat=np.min(lats),            urcrnrlon=np.max(lons),urcrnrlat=np.max(lats),            projection='merc',resolution='l')

m.drawcoastlines(linewidth=2)
m.fillcontinents(color='gray')
m.drawcountries(linewidth=1)

plons, plats = np.meshgrid(lons, lats)
x, y = m(plons, plats)
cp = m.pcolor(x,y,tos[12,:,:])
cbar = plt.colorbar(cp)
cbar.set_label("Temperature [K]")
plt.title("Sea surface temperature")
plt.show()



