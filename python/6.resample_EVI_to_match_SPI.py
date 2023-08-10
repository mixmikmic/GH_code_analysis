import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn

get_ipython().magic('matplotlib inline')
seaborn.set_style('dark')
from scipy import stats
import datetime as dt

ds=xr.open_mfdataset(
    '/g/data/oe9/user/rg6346/VI_mask_nc/EVI/EVI_????.nc') 
ds=ds.rename({'ndvi_evi':'evi'})
ds=ds.drop('band')
ds['evi'] = ds.evi/10000

ds

EVI_month=ds.evi.resample(time='1M').mean(dim='time')
# ndvi_jja=ndvi_q.where(ndvi_q['time.season']=='JJA',drop=True)
# ndvi_djf=ndvi_q.where(ndvi_q['time.season']=='DJF',drop=True)

EVI_month

EVI_month.time[-1]

spi_1M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_1M_masked.nc')
spi_1M

spi_3M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_3M_masked.nc')
spi_3M

spi_6M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_6M_masked.nc')
spi_6M

spi_12M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_12M_masked.nc')
spi_12M

spi_24M= xr.open_dataarray('/g/data/oe9/project/team-drip/Rainfall/SPI_awap/SPI_24M_masked.nc')
spi_24M

spi_1M_sub=spi_1M.isel(time=range(1,204))
spi_1M_sub

EVI_month['time']=spi_1M_sub.time.values

EVI_month

spi_1M_sub.isel(time=range(10,12)).plot.imshow(col='time', robust = True, cmap = 'RdYlGn')

EVI_month.isel(time=range(10,12)).plot.imshow(col='time', robust = True, col_wrap=2, cmap = 'RdYlGn')

# TODO: use scipy.image.ndzoom to upscale soil moisture
from scipy.ndimage import zoom as ndzoom
from scipy.misc import imresize as ndresize
# help(ndzoom)

shape=coarse_EVI.isel(time=0).shape
type(shape)

data = EVI_month.sel(time=coarse_EVI.time[0])
data
out=scipy.misc.imresize(np.nan_to_num(data), shape, interp='cubic',mode = 'F')
out

xr.DataArray(out).plot.imshow(cmap = 'RdYlGn')

from copy import deepcopy

coarse_EVI = deepcopy(spi_1M_sub)
coarse_EVI.attrs = EVI_month.attrs
coarse_EVI.name = EVI_month.name
coarse_EVI['time'] = EVI_month.time

# SPLINE_ORDER = 3  # Try some other values 0 to 5 for SPLINE_ORDER to see what happens



# ZOOM_FACTOR = (len(spi_1M_sub.latitude)/len(EVI_month.lat),
#                len(spi_1M_sub.longitude)/len(EVI_month.long) )

get_ipython().run_cell_magic('time', '', "# It's also possible to zoom a 3D array by setting a factor of 1 for the time\n# dimension, but this way we preserve the timesteps correctly.\nfor timestamp in coarse_EVI.time:\n    # Start by selecting the timestamp\n    data = EVI_month.sel(time=timestamp)\n    # Then zoom to the desired scale, filling nodata values with zero so we can zoom\n    output = ndresize(np.nan_to_num(data), shape, interp='cubic',mode = 'F')\n    # Assign output to the contents of the fine_moisture array\n    coarse = coarse_EVI.sel(time=timestamp)\n    coarse[:] = output\n    \n    # Make sure the minimum is zero, so it remains physically plausible\n    coarse.values[coarse.values < 0] = 0\n    # Last, we'll copy both sets of NaN values so that we don't cause spurious correlations\n    # Try commenting each of these out to see how the map changes!\n    coarse.values[np.isnan(spi_1M_sub.sel(time=timestamp).values)] = np.nan  # from the high-res data\n#     coarse.values[ndzoom(np.isnan(data), zoom=ZOOM_FACTOR, order=0)] = np.nan  # from low-res, with nearest (blocky) zooming\n\ncoarse_EVI.isel(time=0).plot.imshow(robust=True)")

coarse_EVI

spi_1M_sub

# save the coarse_NDVI into a nc file 
path = '/g/data/oe9/project/team-drip/resampled_NDVI/coarse_EVI.nc'
coarse_EVI.to_netcdf(path, mode = 'w')

# Try a few different values to see if the relationship holds
_lat, _lon = 150, 200

# To plot two lines on the same axes, we have to explicitly create and use a set of axes 
# For the second, `ax.twinx()` creates a clone of the axes with a shared x and independent y.
fig, ax = plt.subplots()
coarse_EVI.isel(latitude=100,longitude=120).plot(ax=ax)
spi_1M_sub.isel(latitude=_lat, longitude=_lon).plot(ax=ax.twinx(), color='red')


# Create an empty dataframe to hold our columns of data
df = pd.DataFrame()
# Convert each data array into a series, and add it to the dataframe
for data in [coarse_EVI, spi_1M_sub]:
    df[data.name] = data.to_series()
# Discard any rows with missing values - I would usually keep them,
# but we can't correlate anything with missing data
df = df.dropna()

# And examine the first five rows
df.head(5)

# Try sample numbers between 1 and 100,000; or even delete ".sample()"
df.sample(1000).plot.scatter(x='SPI_1M', y='evi')


seaborn.jointplot(
    x='SPI_1M',
    y='evi',
    data=df.sample(1000),
    # There are several ways to represent a join distribution.
    # Try un-commenting one kind at a time!
    # kind='hex', joint_kws=dict(gridsize=30),
    kind='kde', cmap='magma_r', n_levels=200,
    # kind='scatter',
)



