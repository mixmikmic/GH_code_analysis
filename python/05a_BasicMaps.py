import sys
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import xarray as xr

from cartopy import config
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature 
# When data are defined in lat/lon coordinate system, PlateCarree()
# is the appropriate choice:

from cartopy.util import add_cyclic_point

# get the path of the file. It can be found in the repo data directory.
fname = os.path.join(config["repo_data_dir"],
                     'netcdf', 'HadISST1_SST_update.nc'
                     )

dataset = netcdf_dataset(fname)
sst = dataset.variables['sst'][0, :, :]
lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]
data_crs = ccrs.PlateCarree()   # since our data is on a rectangular lon,lat grid

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, sst, 60,
             transform=ccrs.PlateCarree(), cmap = "jet")
gl = ax.gridlines(crs=data_crs, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlines = False

ax.coastlines()

plt.figure()
ax2 = plt.axes(projection=ccrs.Robinson())
plt.contourf(lons, lats, sst, 60, transform=data_crs, cmap = "jet")
ax2.coastlines()

# A rotated pole projection again...
projection = ccrs.RotatedPole(pole_longitude=-177.5, pole_latitude=37.5)
ax = plt.axes(projection=projection)
ax.set_global()
ax.coastlines()

# ...but now using the transform argument
sst_cyc, lons_cyc = add_cyclic_point(sst, coord=lons)
ax.contourf(lons_cyc, lats, sst_cyc, 60, transform=data_crs, cmap = "jet")

airtemps = xr.tutorial.load_dataset('air_temperature')
sst_cyc, lons_cyc = add_cyclic_point(sst, coord=lons)

plt.figure(figsize=(20, 12))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
im = ax.contourf(lons_cyc, lats, sst_cyc, 60, vmin = -2, vmax = 32, transform=data_crs, cmap = "gist_rainbow_r")
cb = plt.colorbar(im, orientation='vertical', shrink = 0.68, label=r'$\degree C$')
#ax.stock_img()
ax.set_xticks([-120, -60 ,0, 60, 120], crs=ccrs.PlateCarree())
ax.set_yticks([-80,-40, 0, 40, 80], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.gridlines()
ax.add_feature(cfeature.OCEAN, alpha=0.3)
ax.add_feature(cfeature.COASTLINE,linewidth=2)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=1.0)
ax.add_feature(cfeature.RIVERS)
plt.title(r'Our map of $T_s$ ($\degree C$)', fontsize=20)
plt.savefig('WorldMap.png')



