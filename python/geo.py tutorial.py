get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pySW4 as sw4
asterdir = './'

tif = sw4.utils.read_GeoTIFF(asterdir + '/ASTGTM2_N32E035/ASTGTM2_N32E035_dem.tif')
print tif

plt.imshow(tif.elev, cmap='terrain')

plt.imshow(tif.elev, cmap='terrain', extent=tif.extent)

tif.reproject(epsg=32636)
print tif
plt.imshow(tif.elev, cmap='terrain', extent=tif.extent)

tif.resample(to=100)
print tif
plt.imshow(tif.elev, cmap='terrain', extent=tif.extent)

tif.write_GeoTIFF('utm.36.100m.tif')

# read the saved file and plot to see the data
tif = sw4.utils.read_GeoTIFF('utm.36.100m.tif')
print tif
plt.imshow(tif.elev, cmap='terrain', extent=tif.extent)

w,e,s,n = 31, 37, 28, 34
mosaic = sw4.utils.get_dem(asterdir, w,e,s,n)
mosaic.resample(by=0.01)
print mosaic
plt.imshow(mosaic.elev, cmap='terrain', extent=mosaic.extent)

fig, ax = plt.subplots(figsize=(8,8))
m = Basemap(llcrnrlon=w, llcrnrlat=s,
            urcrnrlon=e, urcrnrlat=n,
            resolution='h', projection='lcc',
            lon_0=0.5*(w+e), lat_0=0.5*(s+n),
            area_thresh=1000, ax=ax)

dh = 100.
nx = int((m.xmax-m.xmin)/dh)+1; ny = int((m.ymax-m.ymin)/dh)+1
data = m.transform_scalar(mosaic.elev[::-1],mosaic.x,mosaic.y[::-1],nx,ny)
im = m.imshow(data,cmap='terrain')
m.colorbar(im,"right", size=0.1, pad=0.4)

m.drawcoastlines(color='w')
m.drawcountries(color='w')

parallels = np.arange(s,n,1.)
m.drawparallels(parallels,labels=[1,1,0,0])
meridians = np.arange(w,e,1.)
m.drawmeridians(meridians,labels=[0,0,0,1])

ax.set_title('ASTER GDEM Basemap')

