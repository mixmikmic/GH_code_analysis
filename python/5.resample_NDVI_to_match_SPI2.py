import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn
get_ipython().magic('matplotlib inline')
seaborn.set_style('dark')
from scipy import stats
import datetime as dt

get_ipython().magic('matplotlib inline')

ds=xr.open_mfdataset(
    '/g/data/oe9/user/rg6346/VI_mask_nc/NDVI/NDVI_????.nc') 
ds=ds.rename({'ndvi_evi':'ndvi'})
ds=ds.drop('band')
ds['ndvi'] = ds.ndvi/10000
ds

NDVI_month=ds.ndvi.resample(time='1M').mean(dim='time')
# ndvi_jja=ndvi_q.where(ndvi_q['time.season']=='JJA',drop=True)
# ndvi_djf=ndvi_q.where(ndvi_q['time.season']=='DJF',drop=True)

NDVI_month

NDVI_month.time[-1]

spi_1M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_1M_masked.nc')
spi_1M

spi_1M_sub=spi_1M.isel(time=range(1,204))
spi_1M_sub

NDVI_month['time']=spi_1M_sub.time.values

NDVI_month

spi_1M_sub.isel(time=range(10,12)).plot.imshow(col='time', robust = True, cmap = 'RdYlGn')

# NDVI_month.isel(time=range(10,12)).plot.imshow(col='time', robust = True, col_wrap=2, cmap = 'RdYlGn')

# TODO: use scipy.image.ndzoom to upscale soil moisture
from scipy.ndimage import zoom as ndzoom
from scipy.misc import imresize as ndresize
# help(ndzoom)

# from copy import deepcopy

# fine_spi = deepcopy(NDVI_month)
# fine_spi.attrs = spi_1M_sub.attrs
# fine_spi.name = spi_1M_sub.name
# fine_spi['time'] = spi_1M_sub.time

# SPLINE_ORDER = 3  # Try some other values 0 to 5 for SPLINE_ORDER to see what happens



# ZOOM_FACTOR = (len(NDVI_month.lat) / len(spi_1M_sub.latitude),
#                len(NDVI_month.long) / len(spi_1M_sub.longitude))

from copy import deepcopy

coarse_NDVI= deepcopy(spi_1M_sub)
coarse_NDVI.attrs = NDVI_month.attrs
coarse_NDVI.name = NDVI_month.name
coarse_NDVI['time'] = NDVI_month.time

SPLINE_ORDER = 3  # Try some other values 0 to 5 for SPLINE_ORDER to see what happens



ZOOM_FACTOR = (len(spi_1M_sub.latitude)/len(NDVI_month.lat),
               len(spi_1M_sub.longitude)/len(NDVI_month.long) )

shape=coarse_NDVI.isel(time=0).shape
type(shape)

get_ipython().run_cell_magic('time', '', "# It's also possible to zoom a 3D array by setting a factor of 1 for the time\n# dimension, but this way we preserve the timesteps correctly.\nfor timestamp in coarse_NDVI.time:\n    # Start by selecting the timestamp\n    data = NDVI_month.sel(time=timestamp)\n    # Then zoom to the desired scale, filling nodata values with zero so we can zoom\n    output = ndresize(np.nan_to_num(data), shape, interp='cubic',mode = 'F')\n    # Assign output to the contents of the fine_moisture array\n    coarse = coarse_NDVI.sel(time=timestamp)\n    coarse[:] = output\n    \n    # Make sure the minimum is zero, so it remains physically plausible\n    coarse.values[coarse.values < 0] = 0\n    # Last, we'll copy both sets of NaN values so that we don't cause spurious correlations\n    # Try commenting each of these out to see how the map changes!\n    coarse.values[np.isnan(spi_1M_sub.sel(time=timestamp).values)] = np.nan  # from the high-res data\n    #coarse.values[ndzoom(np.isnan(data), zoom=ZOOM_FACTOR, order=0)] = np.nan  # from low-res, with nearest (blocky) zooming\n\ncoarse_NDVI.isel(time=-1).plot.imshow(robust=True)")

NDVI_month

coarse_NDVI

spi_1M_sub

# save the coarse_NDVI into a nc file 
path = '/g/data/oe9/project/team-drip/resampled_NDVI/coarse_NDVI.nc'
coarse_NDVI.to_netcdf(path, mode = 'w')

# Try a few different values to see if the relationship holds
_lat, _lon = 100, 120

# To plot two lines on the same axes, we have to explicitly create and use a set of axes 
# For the second, `ax.twinx()` creates a clone of the axes with a shared x and independent y.
fig, ax = plt.subplots()
coarse_NDVI.isel(latitude=100,longitude=120).plot(ax=ax)
spi_1M_sub.isel(latitude=_lat, longitude=_lon).plot(ax=ax, color='red')


# Create an empty dataframe to hold our columns of data
df = pd.DataFrame()
# Convert each data array into a series, and add it to the dataframe
for data in [coarse_NDVI, spi_1M_sub]:
    df[data.name] = data.to_series()
# Discard any rows with missing values - I would usually keep them,
# but we can't correlate anything with missing data
df = df.dropna()

# And examine the first five rows
df.head(5)

# Try sample numbers between 1 and 100,000; or even delete ".sample()"
df.sample(1000).plot.scatter(x='SPI_1M', y='ndvi')


seaborn.jointplot(
    x='SPI_1M',
    y='ndvi',
    data=df.sample(1000),
    # There are several ways to represent a join distribution.
    # Try un-commenting one kind at a time!
    #kind='hex', joint_kws=dict(gridsize=30),
    # kind='kde', cmap='magma_r', n_levels=200,
    kind='scatter',
)



