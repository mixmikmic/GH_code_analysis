import numpy as np
import xarray as xr

ds = xr.tutorial.load_dataset('rasm')
da = ds['Tair']

ds

da.mean(dim=('x', 'y'))

# dataarray + scalars
da - 273.15  # (K --> C)

da_mean = da.mean(dim='time')
da_mean

# dataarray + dataarray
da - da_mean  

# Notice that this required broadcasting along the time dimension

# Using groupby to calculate a monthly climatology:

da_climatology = da.groupby('time.month').mean('time')

da_climatology

roller = da.rolling(time=3)
roller

roller.mean()

# we can also provide a custom function 

def sum_minus_2(da, axis):
    return da.sum(axis=axis) - 2

roller.reduce(sum_minus_2)

da_noise = da + np.random.random(size=da.shape)
da_noise

# some example legacy code to calculate the spearman correlation coefficient

import bottleneck


def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

def correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

def spearman_correlation_gufunc(x, y):
    x_ranks = bottleneck.rankdata(x, axis=-1)
    y_ranks = bottleneck.rankdata(y, axis=-1)
    return correlation_gufunc(x_ranks, y_ranks)

# Wrap the "legacy code" with xarray's apply_ufunc. 
def spearman_correlation(x, y, dim):
    return xr.apply_ufunc(
        spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

da_corr = corr = spearman_correlation(da, da_noise, 'time')
da_corr

# mask out 1's in the correlation array

da_corr.where(da_corr < 1)

# xarray also provides a function for where
xr.where(da_corr > 0.996, 0, 1)



