get_ipython().magic('matplotlib inline')

import numpy as np
import xarray as xr
import pandas as pd

start = pd.Timestamp('2015-01-01')
end = pd.Timestamp('2015-07-01')
t = np.linspace(start.value, end.value, 10)
arr = np.random.rand(1000, 1000, 10)
xa1 = xr.DataArray(arr, dims=('x', 'y', 'time'), coords={'time':pd.DatetimeIndex(t)}, name='data')

start = pd.Timestamp('2015-07-01')
end = pd.Timestamp('2016-01-01')
t = np.linspace(start.value, end.value, 10)
arr = np.random.rand(1000, 1000, 10) * 2.5
xa2 = xr.DataArray(arr, dims=('x', 'y', 'time'), coords={'time':pd.DatetimeIndex(t)}, name='data')

xa1.to_netcdf('DaskTest1.nc')

xa2.to_netcdf('DaskTest2.nc')



data = xr.open_mfdataset(['DaskTest1.nc', 'DaskTest2.nc'], chunks={'time':10})['data']

m = data.mean(dim='time')

seasonal = data.groupby('time.season').mean(dim='time')

from dask.dot import dot_graph

dot_graph(m.data.dask)

dot_graph(seasonal.data.dask)



