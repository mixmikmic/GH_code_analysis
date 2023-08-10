get_ipython().magic('matplotlib inline')
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ghcncams/air.mon.mean.nc'

ds = xr.open_dataset(url)

var='air'
dvar = ds[var]
dvar

# slice a longitude range, a latitute range and a specific time value
lat_bnds, lon_bnds = [50, 18], [-130+360., -64+360.]
ds_cut = ds.sel(lat=slice(*lat_bnds), lon=slice(*lon_bnds), time='2015-03-01')
ds_cut

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# make a nice date string for titling the plot
date_string = pd.Timestamp(ds_cut.time.data).strftime('%B %Y')

# mask NaN values and convert Kelvin to Celcius
t_c = np.ma.masked_invalid(ds_cut.air)-272.15

# PlateCarree is rectilinear lon,lat
data_crs = ccrs.PlateCarree()
# Albers projection for the continental US
plot_crs = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=23.)

#Cartopy can use features from naturalearthdata.com
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

# plot using Cartopy
# using Albers projection with coastlines, country and state borders
fig = plt.figure(figsize=(12,8))
ax = plt.axes(projection=plot_crs)
mesh = ax.pcolormesh(ds_cut.lon, ds_cut.lat, t_c, transform=data_crs, zorder=0, vmin=-10, vmax=30)
fig.colorbar(mesh)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray')
ax.set_title('{}: {}'.format(dvar.long_name, date_string));

