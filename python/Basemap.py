# after getting data
import netCDF4 as nc

f = nc.Dataset('GEOS.fp.fcst.inst3_2d_met_Nx.20161027_00+20161027_0000.V01.nc4', 'r')
print(f.variables['CLDHGH'])
print(f.variables['CLDHGH'].shape)
f.close()

# let's plot!
get_ipython().magic('matplotlib inline')
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

f = nc.Dataset('GEOS.fp.fcst.inst3_2d_met_Nx.20161027_00+20161027_0000.V01.nc4', 'r')
data = f.variables['CLDHGH']
lon = f.variables['lon']
lat = f.variables['lat']
# create 2d grid
lon, lat = np.meshgrid(lon, lat)
fig = plt.figure()

m = Basemap(projection='kav7',lon_0=0,resolution=None)
m.drawmapboundary(fill_color='0.3')
im1 = m.pcolormesh(lon,lat,data[0],shading='flat',cmap=plt.cm.gist_gray,latlon=True)
m.drawparallels(np.arange(-90.,99.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))

m = Basemap(projection='mill',lon_0=0)
m.drawmapboundary(fill_color='aqua')
im1 = m.pcolormesh(lon,lat,data[0],shading='flat',cmap=plt.cm.gray,latlon=True)
m.fillcontinents()
m.drawparallels(np.arange(-90.,99.,30.))
m.drawmeridians(np.arange(-180.,180.,60.))

# let's make this pretty
f.variables['CLDHGH'].missing_value
data = np.ma.masked_where(data > f.variables['CLDHGH'].missing_value, data)
data

m = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m.bluemarble()

