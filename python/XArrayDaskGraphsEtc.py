import numpy as np
import xarray as xr
from datetime import datetime
from dask.dot import dot_graph

data = np.random.rand(100, 100)

b = xr.DataArray(data, dims=['x', 'y'], coords={'time':datetime(2010, 1, 4)})
c = xr.DataArray(data*2, dims=['x', 'y'], coords={'time':datetime(2010, 1, 5)})
d = xr.DataArray(data*3, dims=['x', 'y'], coords={'time':datetime(2010, 1, 6)})

a = xr.concat([b, c, d], 'time')

ds = a.to_dataset(name='data')

ds.to_netcdf('XArrayExample.nc')

data = xr.open_dataset('XArrayExample.nc', chunks={'x': 50, 'y': 50})['data']

data

data.chunks

doubled = data * 2

# We have to use the .data.dask attribute to get the graph when we're using dask through xarray
dot_graph(doubled.data.dask)

min_doubled = (data * 2).min()

dot_graph(min_doubled.data.dask)

mean_doubled = (data * 2).mean()

dot_graph(mean_doubled.data.dask)

# Just chunk in the y direction
data = xr.open_dataset('XArrayExample.nc', chunks={'y': 50})['data']

mean_doubled = (data * 2).mean()
dot_graph(mean_doubled.data.dask)

data = xr.open_dataset('XArrayExample.nc', chunks={'x': 10, 'y': 30})['data']

mean_doubled = (data * 2).mean()
dot_graph(mean_doubled.data.dask)

data = xr.open_dataset('XArrayExample.nc', chunks={'x': 50, 'y': 50})['data']

ts = data.isel(x=5, y=5)

ts = ts * 2

dot_graph(ts.data.dask)

data = np.random.rand(10000, 10000)

b = xr.DataArray(data, dims=['x', 'y'], coords={'time':datetime(2010, 1, 4)})
c = xr.DataArray(data*2, dims=['x', 'y'], coords={'time':datetime(2010, 1, 5)})
d = xr.DataArray(data*3, dims=['x', 'y'], coords={'time':datetime(2010, 1, 6)})

a = xr.concat([b, c, d], 'time')

ds = a.to_dataset(name='data')

ds.to_netcdf('XArrayExample_Large.nc')

from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

data = xr.open_dataset('XArrayExample_Large.nc', chunks={'x': 5000, 'y': 5000})['data']

res = (data * 2).mean()

with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
        res.load()

from dask.diagnostics import visualize
b = visualize([prof, rprof, cprof])

