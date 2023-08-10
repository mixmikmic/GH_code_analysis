from siphon.ncss import NCSS

subset = NCSS('http://thredds.cencoos.org/thredds/ncss/'
              'G1_SST_US_WEST_COAST.nc')

[value for value in dir(subset) if not value.startswith('__')]

[value for value in dir(subset.metadata) if not value.startswith('__')]

subset.metadata.time_span

subset.metadata.lat_lon_box

subset.variables

subset.metadata.variables

query = subset.query()

query.lonlat_box(east=-120, north=50, south=35, west=-135)

from datetime import datetime

query.time(datetime(2013, 12, 31))

variable = 'analysed_sst'

query.variables(variable)

query.spatial_query, query.time_query

import time

start_time = time.time()

data = subset.get_data(query)

elapsed = time.time() - start_time
print('{:.2f} s'.format(elapsed))

type(data)

data.filepath()

import numpy.ma as ma

lon = data['lon'][:]
lat = data['lat'][:]

temp = data[variable][:]
temp = ma.masked_invalid(temp.squeeze())

time = data['time']
time = ''.join(time[:][0].astype(str).tolist())

lon[0], lon[-1], lat[0], lat[-1], time

get_ipython().magic('matplotlib inline')

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from cmocean import cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

fig, ax = plt.subplots(figsize=(9, 9),
                       subplot_kw=dict(projection=ccrs.PlateCarree()))

ax.coastlines(resolution='50m')
cs = ax.pcolormesh(lon, lat, temp, cmap=cm.thermal)
cbar = fig.colorbar(cs, extend='both', shrink=0.75)

gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

