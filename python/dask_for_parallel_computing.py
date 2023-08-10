import numpy as np
shape = (1000, 4000)
ones_np = np.ones(shape)
ones_np

ones_np.nbytes / 1e6

import dask.array as da
ones = da.ones(shape)

chunk_shape = (1000, 1000)
ones = da.ones(shape, chunks=chunk_shape)
ones

ones.compute()

ones.visualize()

sum_of_ones = ones.sum()
sum_of_ones.visualize()

fancy_calculation = (ones * ones[::-1, ::-1]).mean()
fancy_calculation.visualize()

bigshape = (200000, 4000)
big_ones = da.ones(bigshape, chunks=chunk_shape)
big_ones

big_ones.nbytes / 1e6

from dask.diagnostics import ProgressBar

big_calc = (big_ones * big_ones[::-1, ::-1]).mean()

with ProgressBar():
    result = big_calc.compute()
result

big_ones_reduce = (np.cos(big_ones)**2).mean(axis=0)
big_ones_reduce

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12,8)

plt.plot(big_ones_reduce)

from dask.distributed import Client, LocalCluster

lc = LocalCluster(n_workers=1)
client = Client(lc)
client

big_calc.compute()

random_values = da.random.normal(size=(2e8,), chunks=(1e6,))
hist, bins = da.histogram(random_values, bins=100, range=[-5, 5]) 

hist

x = 0.5 * (bins[1:] + bins[:-1])
width = np.diff(bins)
plt.bar(x, hist, width);

get_ipython().system(' curl -O http://www.ldeo.columbia.edu/~rpa/aviso_madt_2015.tar.gz')

get_ipython().system(' tar -xvzf aviso_madt_2015.tar.gz')

get_ipython().system(' ls 2015 | wc -l')

import xarray as xr
xr.__version__

ds_first = xr.open_dataset('2015/dt_global_allsat_madt_h_20150101_20150914.nc')
ds_first

ds_first.nbytes / 1e6

help(xr.open_mfdataset)

# On I got a "Too many open files" OSError.
# It's only 365 files. That shouldn't be too many. 
# However, I discovered my ulimit was extremely low.
# One workaround is to call 
#  $ ulimit -S -n 4000
# from the command line before launching the notebook

ds = xr.open_mfdataset('2015/*.nc')
ds

ssh = ds.adt
ssh

ssh[0].plot()

ssh_2015_mean = ssh.mean(dim='time')
ssh_2015_mean.load()

ssh_2015_mean.plot()

ssh_anom = ssh - ssh_2015_mean
ssh_variance_lonmean = (ssh_anom**2).mean(dim=('lon', 'time'))

ssh_variance_lonmean.plot()

weight = np.cos(np.deg2rad(ds.lat))
weight /= weight.mean()
(ssh_anom * weight).mean(dim=('lon', 'lat')).plot()



