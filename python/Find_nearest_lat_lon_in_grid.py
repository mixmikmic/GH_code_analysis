get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Latitudes
longitude = np.arange(-130, -65, .5)

# Longitudes
latitude = np.arange(20, 55, .5)

# Use meshgrid to make 2D arrays of the lat/lon data
lats, lons = np.meshgrid(latitude, longitude)

# A 2D numpy array of your temperatures, or some other variable, that matches the lat/lon grid
# Generate randomized test data for now
data = np.random.randint(50, size=np.shape(lats))

# Stations latitude longitude, the nearest gridpoint we are looking for
stn_lat = 40.3242
stn_lon = -110.2962

# Create a basemap to show the data
m = Basemap(llcrnrlon=longitude[0],llcrnrlat=latitude[0],urcrnrlon=longitude[-1],urcrnrlat=latitude[-1],)
m.drawcoastlines()
m.pcolormesh(lons, lats, data, alpha=.3)
m.scatter(stn_lon, stn_l)

abslat = np.abs(lats-stn_lat)
abslon= np.abs(lons-stn_lon)

m.drawcoastlines()
m.pcolormesh(lons, lats, abslat)

m.drawcoastlines()
m.pcolormesh(lons, lats, abslon)

c = np.maximum(abslon, abslat)

latlon_idx = np.argmin(c)

m.drawcoastlines()
m.pcolormesh(lons, lats, c)

print latlon_idx

grid_temp = data.flat[latlon_idx]

x, y = np.where(c == np.min(c))
grid_data = data[x[0], y[0]]
grid_lat = lats[x[0], y[0]]
grid_lon = lons[x[0], y[0]]

print "Value of %s at %s %s" % (grid_data, grid_lat, grid_lon)
print "%s %s is the nearest grid to %s %s" % (grid_lat, grid_lon, stn_lat, stn_lon)

m.drawcoastlines()
m.pcolormesh(lons, lats, c)
m.scatter(grid_lon, grid_lat, s=100, c='w')
plt.text(grid_lon+.5, grid_lat+.5, grid_data, color='w')



