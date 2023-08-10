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

ddir = '/rigel/ocp/users/ra2697/channel_topography/GCM/run_taux2000_rb0110_bump'
get_ipython().magic("prun ds = open_mdsdataset(ddir, prefix=['U', 'Eta', 'T', 'V', 'S', 'W'])")
# 3 minutes

ds

plt.figure(figsize=(12,10))
ds['T'].isel(time=0, Z=0).plot()

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

# take the mean of the whole dataset
get_ipython().magic('time dsm = (dsp**2).mean().compute()')
dsm

# rechunk in time 
dspr = dsp.chunk({'time': len(dsp.time),
# crashed
#                  'Z': 1, 'Zl':1, 'Zp1': 1,
                   'YC': 1, 'YG': 1
                 }).persist()
#dspr.chunks

dspr

# 2D cross spectrum
import dask.array

w_fft = dask.array.fft.fftn(dspr.W.data, axes=(0,3))
w_2d_ps = (w_fft * w_fft.conj()).real.mean(axis=2)
w_2d_psc = w_2d_ps.compute()

v_fft = dask.array.fft.fftn(dspr.V.data, axes=(0,3))
v_2d_ps = (v_fft * v_fft.conj()).real.mean(axis=2)
v_2d_psc = v_2d_ps.compute()

t_fft = dask.array.fft.fftn(dspr['T'].data, axes=(0,3))
wt_2d_cross_spec = (t_fft * w_fft.conj()).real.mean(axis=2).compute()

vt_2d_cross_spec = (t_fft * v_fft.conj()).real.mean(axis=2).compute()

from matplotlib.colors import SymLogNorm

plt.pcolormesh(np.fft.fftshift(wt_2d_cross_spec[:,5], axes=(0,1)), cmap='RdBu_r',
               norm=SymLogNorm(0.1, vmin=-1e3, vmax=1e3))
plt.colorbar()

plt.pcolormesh(np.fft.fftshift(wt_2d_cross_spec[:,15], axes=(0,1)), cmap='RdBu_r',
               norm=SymLogNorm(0.1, vmin=-1e3, vmax=1e3))
plt.colorbar()

plt.pcolormesh(np.fft.fftshift(vt_2d_cross_spec[:,5], axes=(0,1)), cmap='RdBu_r',
               norm=SymLogNorm(1e3, vmin=-1e7, vmax=1e7))
plt.colorbar()

plt.pcolormesh(np.fft.fftshift(vt_2d_cross_spec[:,15], axes=(0,1)), cmap='RdBu_r',
               norm=SymLogNorm(1e3, vmin=-1e7, vmax=1e7))
plt.colorbar()

plt.pcolormesh(np.log10(np.fft.fftshift(w_2d_psc[:,10], axes=(0,1))))

plt.pcolormesh(np.log10(np.fft.fftshift(v_2d_psc[:,10], axes=(0,1))))

# take the Fourier transform along one dimension
v_fft = xrft.dft(ds.V, dim=['XC'], shift=False)
# multiply by complex conjugate
pow_spec = (v_fft*v_fft.conj()).mean(dim=('YG','time'))
pow_spec = pow_spec.astype('f8').rename('V_power_spectrum')
pow_spec

ds['V'].nbytes / 1e9 # data size in Gigabytes (GB)

print(client)
get_ipython().magic('time pow_spec.load()')

5*60+33

plt.rcParams['font.size'] = 12
plt.figure(figsize=(10,8))
# timing data
timing = pd.DataFrame([
    (1,24,123.), # data not cached
    (1,24,53.),
    (1,24,102.),
    (1,24,53.),
    (1,12,139.),
    (1,12,114.),
    (1,12,70.),
    (1,6,198),
    (1,6,123),
    (2,48,77.),
    (2,48,28.),
    (2,48,28.1),
    (5,120,178.),
    (5,120,13.),
    (5,120,12.7),
    (5,120,12.8),
    (5,120,13.5),
    (5,120,61.),
    (5,120,15.5),
    (5,120,54.4),
    # these are from 3/31/2017, something was wrong with the cluster
    (5,120,333)
    ],columns=('nprocs', 'ncores', 'time'))
#  (MB/s)
timing['throughput'] = (ds['V'].nbytes /1e6)/ timing['time']
timing.groupby('ncores')['throughput'].max().plot(marker='o', label='fastest')
timing.groupby('ncores')['throughput'].min().plot(marker='o', label='slowest')
plt.legend(loc='upper left')
plt.title('Processing Throughput')
plt.xlabel('number of cores')
plt.ylabel('MB/s')

import dask.array as da

import xgcm

grid = xgcm.Grid(dsp)
grid

vort3 = (-grid.diff(dsp.U*dsp.dxC, 'Y') + grid.diff(dsp.V*dsp.dyC, 'X'))/dsp.rAz
vort3

fig, ax = plt.subplots(figsize=(12,10))
vort3[0,0].plot(vmax=1e-4, add_colorbar=False)

vort3_data = vort3.isel(Z=0).data
#vort3_data_flat = vort3_data.reshape((nt, (ny*nx)))
vort3_hist, bins = da.histogram(vort3.isel(Z=0).data, bins=np.arange(-7e-4,7e-4,1e-5))

get_ipython().magic('time vort3_hist_, bins_ = da.compute(vort3_hist, bins)')

# it might just be the rolling that causes the histogram to be slow
get_ipython().magic('time (vort3_data**2).mean().compute()')

fig, ax = plt.subplots(figsize=(12,10))
ax.bar(bins_[:-1], vort3_hist_, width=bins_[1]-bins[0])
ax.set_yscale('log')

# can dask do it faster?
uvel = ds.U.data
vvel = ds.V.data
# no! dask has no roll
#vort_toy = -(uvel - ) + (vvel - da.roll(vvel, -1))
#%time (vort_toy**2).mean().compute()

# test of rolling speed
delta_u = grid.diff(ds.U, 'Y')
du2_mean = (delta_u.isel(Z=0)**2).mean()
get_ipython().magic('time du2_mean.compute()')

eta_hist, eta_bins = da.histogram(ds.Eta.data, bins=np.arange(-1.,1.,0.1))
eta_hist_, eta_bins_ = da.compute(eta_hist, eta_bins)

fig, ax = plt.subplots(figsize=(12,10))
ax.bar(eta_bins_[:-1], eta_hist_, width=eta_bins_[1]-eta_bins[0])

# something more computational: SVD
eta_data = ds.Eta.data
nt, ny, nx = eta_data.shape
eta_data_flat = eta_data.reshape((nt, (ny*nx)))
eta_data_flat

u, s, v = da.linalg.svd_compressed(eta_data_flat, 5, n_power_iter=2)
#u, s, v = da.linalg.svd(eta_data_flat)

print(u.shape, s.shape, v.shape)

get_ipython().magic('time u_, s_, v_ = da.compute(u, s, v)')

v_rs = v_.reshape((5,ny,nx))

plt.pcolormesh(v_rs[1])

plt.plot(s_)

data = np.random.rand(10000,1000)

get_ipython().magic('time futures = client.scatter([data])')
# with en0: 2.88 s
# with ib0: 1.7s

get_ipython().magic('time client.gather(futures)')
# with en0: 1.7 s
# with ib0: .5 s

client.restart()



