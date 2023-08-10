indexfile = "datafiles/index_latest.txt"

import numpy as np
dataindex = np.genfromtxt(indexfile, skip_header=6, unpack=True, delimiter=',', dtype=None,               names=['catalog_id', 'file_name', 'geospatial_lat_min', 'geospatial_lat_max',
                     'geospatial_lon_min', 'geospatial_lon_max',
                     'time_coverage_start', 'time_coverage_end', 
                     'provider', 'date_update', 'data_mode', 'parameters'])

lon_min = dataindex['geospatial_lon_min']
lon_max = dataindex['geospatial_lon_max']
lat_min = dataindex['geospatial_lat_min']
lat_max = dataindex['geospatial_lat_max']

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

m = Basemap(projection='merc', llcrnrlat=30., urcrnrlat=46.,
            llcrnrlon=-10, urcrnrlon=40., lat_ts=38., resolution='l')
lonmean, latmean = 0.5*(lon_min + lon_max), 0.5*(lat_min + lat_max)
lon2plot, lat2plot = m(lonmean, latmean)

fig = plt.figure(figsize=(10,8))
m.plot(lon2plot, lat2plot, 'ko', markersize=2)
m.drawcoastlines(linewidth=0.5, zorder=3)
m.fillcontinents(zorder=2)

m.drawparallels(np.arange(-90.,91.,2.), labels=[1,0,0,0], linewidth=0.5, zorder=1)
m.drawmeridians(np.arange(-180.,181.,3.), labels=[0,0,1,0], linewidth=0.5, zorder=1)
plt.show()

box = [12, 15, 32, 34]

import numpy as np
goodcoordinates = np.where( (lonmean>=box[0]) & (lonmean<=box[1]) & (latmean>=box[2]) & (latmean<=box[3]))
print goodcoordinates

goodfilelist = dataindex['file_name'][goodcoordinates]
print goodfilelist

m2 = Basemap(projection='merc', llcrnrlat=32., urcrnrlat=34.,
            llcrnrlon=12, urcrnrlon=15., lat_ts=38., resolution='h')
lon2plot, lat2plot = m2(lonmean[goodcoordinates], latmean[goodcoordinates])

fig = plt.figure(figsize=(10,8))
m2.plot(lon2plot, lat2plot, 'ko', markersize=4)
m2.drawcoastlines(linewidth=0.5, zorder=3)
m2.fillcontinents(zorder=2)

m2.drawparallels(np.arange(-90.,91.,0.5), labels=[1,0,0,0], linewidth=0.5, zorder=1)
m2.drawmeridians(np.arange(-180.,181.,0.5), labels=[0,0,1,0], linewidth=0.5, zorder=1)
plt.show()

