import xarray as xr
from dask.dot import dot_graph
import dask.array as da
import numpy as np
import os
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

import pandas as pd
import xgcm
from xmitgcm import open_mdsdataset
import xrft

import distributed
print(distributed.__version__)
#import blosc
#print(blosc.__version__)
import dask
print(dask.__version__)
print(xr.__version__)

data_dir = '/rigel/ocp/users/db3157/channel_all_res/02km/run_5km_start_tracer'
nsteps = range(4665600, 4665600+103680, (4665744-4665600))
get_ipython().magic("time ds_tracer_2 = open_mdsdataset(data_dir, delta_t=300, iters= nsteps, prefix=['PTRACER01','T','W','V'], ignore_unknown_vars=True, geometry='cartesian')")

ds = ds_tracer_2
ds

ds.nbytes / 1e9

plt.figure(figsize=(12,10))
ds['PTRACER01'].isel(time=100, Z=10).plot()

from dask.distributed import Client

client.restart()

client.shutdown()

#scheduler_file = '../.dask_scheduler/dask_scheduler_file-' + os.environ['SLURM_JOBID']
scheduler_file = '/rigel/home/ra2697/.dask_schedule_file.json'
client = Client(scheduler_file=scheduler_file)
client

# persist dataset, load in to memory
get_ipython().magic('time dsp = ds.persist()')

dsp.nbytes / 1e9

dsp

# just load one slice
ds_slice = dsp.isel(YG=400, YC=400)
ds_slice

ds_slice = ds_slice.load()
# hack the dimensions
ds_slice.W.variable.dims = ['time', 'Z', 'XC']
ds_slice.V.variable.dims = ['time', 'Z', 'XC']
ds_slice

ds_slice['PTRACER01_anom'] = ds_slice.PTRACER01 - ds_slice.PTRACER01.mean(dim='XC')

plt.figure(figsize=(12,8))
ds_slice.PTRACER01_anom.isel(Z=10).plot()

plt.figure(figsize=(12,8))
ds_slice.W.isel(Z=10).plot(vmax=1e-3)

import xrft

W_ps = xrft.power_spectrum(ds_slice.W, dim=['XC', 'time'])
W_ps

plt.figure(figsize=(12,8))
np.log10(W_ps.isel(Z=10)).plot(vmin=0, vmax=3)

wtr_cs = xrft.cross_spectrum(ds_slice.W, ds_slice.PTRACER01_anom, dim=['XC', 'time'])

# smooth it
wtr_cs_sm = xr.DataArray(
                gaussian_filter(wtr_cs, 2.0),
                coords=wtr_cs.coords, dims=wtr_cs.dims)

from matplotlib.colors import LogNorm, SymLogNorm
from scipy.ndimage.filters import gaussian_filter

month = (24*60*60*31)**-1
km100 = (100e3)**-1

fig, ax = plt.subplots(figsize=(12,8))
wtr_cs_sm.isel(Z=7).plot(norm=SymLogNorm(1000), ax=ax)
ax.axhline(month, color='k')
ax.axhline(-month, color='k')
ax.axvline(km100, color='k')
ax.axvline(-km100, color='k')
#ax.set_yscale('symlog', linthreshy=month/12)
#ax.set_xscale('symlog', linthreshx=km100/10)

z = wtr_cs_sm.Z
plt.plot(wtr_cs_sm.sum(dim=['freq_XC', 'freq_time']), z, label='full')
plt.plot(wtr_cs_sm.where(abs(wtr_cs_sm.freq_XC)>km100).sum(dim=['freq_XC', 'freq_time']), z, label='< 100 km')
plt.plot(wtr_cs_sm.where(abs(wtr_cs_sm.freq_time)>month).sum(dim=['freq_XC', 'freq_time']), z, label='< 1 month')
plt.legend()
plt.ylabel('Z (m)')
plt.title('Vertical Tracer Flux')

wtr_cs_sm.isel(Z=7).where(abs(wtr_cs_sm.freq_XC)>km100).plot(norm=SymLogNorm(1000))



