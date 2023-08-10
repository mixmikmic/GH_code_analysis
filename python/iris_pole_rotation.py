get_ipython().magic('matplotlib inline')

import numpy
import matplotlib.pyplot as plt
from iris.analysis.cartography import rotate_pole
import cartopy.crs as ccrs

lons = numpy.arange(-180, 180, 5)
lats = numpy.zeros(lons.shape)

plt.axes(projection=ccrs.PlateCarree(central_longitude=-180))

plt.plot(lons, lats, transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))

plt.gca().set_global()
plt.gca().coastlines()
plt.show()

ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=-180))

plt.plot(numpy.array([115, 225]), numpy.array([10, 10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))
plt.plot(numpy.array([115, 225]), numpy.array([-10, -10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))
plt.plot(numpy.array([115, 115]), numpy.array([-10, 10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))
plt.plot(numpy.array([225, 225]), numpy.array([-10, 10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))

plt.gca().set_global()
plt.gca().coastlines()
plt.show()

lons = numpy.arange(-55, 45, 5)
lats = numpy.zeros(lons.shape)

plt.axes(projection=ccrs.RotatedPole(260, 20, central_rotated_longitude=-180))

plt.plot(lons, lats)

plt.gca().set_global()
plt.gca().coastlines()
plt.show()

ax = plt.axes(projection=ccrs.SouthPolarStereo())

plt.plot(numpy.array([115, 225]), numpy.array([10, 10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))
plt.plot(numpy.array([115, 225]), numpy.array([-10, -10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))
plt.plot(numpy.array([115, 115]), numpy.array([-10, 10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))
plt.plot(numpy.array([225, 225]), numpy.array([-10, 10]), 
         transform=ccrs.RotatedPole(pole_longitude=260, pole_latitude=20))

ax.set_extent((-180, 180, -90.0, -20), crs=ccrs.PlateCarree())
plt.gca().coastlines()
plt.show()



