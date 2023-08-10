import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bruges as b
from obspy.io.segy.segy import _read_segy
from scipy.interpolate import griddata
from scipy.interpolate import splev, splrep

get_ipython().magic('matplotlib inline')
# comment out the following if you're not on a Mac with HiDPI display
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

colr='rainbow'

# load Top Heimdal time interpretation
hrz=np.recfromtxt('Top_Heimdal_subset.txt', names=['il','xl','z'])
# load avo intercept & gradient at Top Heimdal
ig=np.recfromtxt('RG_3dinv.txt', names=['i','g','z','xl','il'], skip_header=16)

# gridding
inl=np.arange(1300,1502,2)
crl=np.arange(1500,2002,2)
xi = np.linspace(inl.min(), inl.max(),250)
yi = np.linspace(crl.min(), crl.max(),250)
X, Y = np.meshgrid(xi, yi)
Z = griddata((hrz['il'], hrz['xl']), hrz['z'], (X,Y), method='cubic')
I=griddata((ig['il'],ig['xl']),ig['i'],(X,Y),method='cubic')
G=griddata((ig['il'],ig['xl']),ig['g'],(X,Y),method='cubic')

segy_near = _read_segy('3d_nearstack.sgy')
segy_far = _read_segy('3d_farstack.sgy')

ns=segy_near.binary_file_header.number_of_samples_per_data_trace
sr=segy_near.binary_file_header.sample_interval_in_microseconds/1000.
near = np.vstack([xx.data for xx in segy_near.traces]).T
near = near.reshape(ns,inl.size,crl.size)
far = np.vstack([xx.data for xx in segy_far.traces]).T
far = far.reshape(ns,inl.size,crl.size)

lagtime=segy_near.traces[0].header.lag_time_A*-1
twt=np.arange(lagtime,sr*ns+lagtime,sr)

hrz_extr=np.zeros((hrz.size,5))
twt_finer=np.arange(hrz['z'].min(), hrz['z'].max(),0.1) # creates twt scale at 0.1 ms scale

for i in range(hrz.size):
    ii_idx=inl.tolist().index(hrz['il'][i])
    cc_idx=crl.tolist().index(hrz['xl'][i])
    zz_idx = np.abs(twt-hrz['z'][i]).argmin()

    trace_near = near[:, ii_idx, cc_idx].flatten()
    trace_far = far[:, ii_idx, cc_idx].flatten()
    amp_near = splev(hrz['z'][i], splrep(twt, trace_near))
    amp_far = splev(hrz['z'][i], splrep(twt, trace_far))

    hrz_extr[i,0] = hrz['il'][i]
    hrz_extr[i,1] = hrz['xl'][i]
    hrz_extr[i,2] = hrz['z'][i]
    hrz_extr[i,3] = amp_near
    hrz_extr[i,4] = amp_far

An=griddata((hrz_extr[:,0],hrz_extr[:,1]),hrz_extr[:,3],(X,Y),method='cubic')
Af=griddata((hrz_extr[:,0],hrz_extr[:,1]),hrz_extr[:,4],(X,Y),method='cubic')

# reshape arrays in this way
# x-axis:  sin^2(ang_near)  sin^2(ang_far)
#          sin^2(ang_near)  sin^2(ang_far)
#          ...
#          sin^2(ang_near)  sin^2(ang_far)

# y-axis:  AMP_near         AMP_far
#          AMP_near         AMP_far
#          ...
#          AMP_near         AMP_far

ang_n=10
ang_f=30
x = np.sin(np.deg2rad([ang_n,ang_f]))**2

dd=An.size # number of rows will be equal to total number of cells of amplitude maps

xx = np.reshape(np.resize(x,dd*2),(dd,2))
yy = np.empty(xx.shape)
yy[:,0] = An.flatten()
yy[:,1] = Af.flatten()

print('*** x-coordinates array; each row is linearized angle for near and far')
print(xx)
print('*** y-coordinates array; each row is the near and far amplitudes for each bin')
print(yy)

import time
from scipy import stats

def multiple_linregress(x, y):
    x_mean = np.mean(x, axis=1, keepdims=True)
    x_norm = x - x_mean
    y_mean = np.mean(y, axis=1, keepdims=True)
    y_norm = y - y_mean
    slope = (np.einsum('ij,ij->i', x_norm, y_norm) /
             np.einsum('ij,ij->i', x_norm, x_norm))
    intercept = y_mean[:, 0] - slope * x_mean[:, 0]
    return np.column_stack((slope, intercept))

avo_I1 = np.empty(An.size)
avo_G1 = np.empty(An.size)
print('****** np.polyfit ******')
start = time.time()
for j in range(An.size):
    y = np.array([An.flatten()[j], Af.flatten()[j]])
    gradient, intercept = np.polyfit(x,y,1)
    avo_I1[j]= intercept
    avo_G1[j]= gradient
end = time.time()
elapsed = end - start
print('Time elapsed: {:.2f} seconds.'.format(elapsed))

avo_I2 = np.empty(An.size)
avo_G2 = np.empty(An.size)
print('****** stats.linregress ******')
start = time.time()
for j in range(An.size):
    y = np.array([An.flatten()[j], Af.flatten()[j]])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    avo_I2[j]= intercept
    avo_G2[j]= slope   
end = time.time()
elapsed = end - start
print('Time elapsed: {:.2f} seconds.'.format(elapsed))

avo_I3 = np.empty(An.size)
avo_G3 = np.empty(An.size)
print('****** multiple_linregress ******')
start = time.time()
out = multiple_linregress(xx,yy)
avo_G3=out[:,0]
avo_I3=out[:,1]
end = time.time()
elapsed = end - start
print('Time elapsed: {:.2f} seconds.'.format(elapsed))

f, ax = plt.subplots(nrows=2,ncols=2, figsize=(8,8))
ax[0,0].hist(avo_I1.ravel(), bins=25, color='b')
ax[0,1].hist(avo_G1.ravel(), bins=25, color='b')
ax[1,0].hist(avo_I3.ravel(), bins=25, color='r')
ax[1,1].hist(avo_G3.ravel(), bins=25, color='r')
ax[0,0].set_title('AVO Intercept')
ax[0,1].set_title('AVO Gradient')

avo_I1=avo_I1.reshape(X.shape)
avo_G1=avo_G1.reshape(Y.shape)
avo_I3=avo_I3.reshape(X.shape)
avo_G3=avo_G3.reshape(Y.shape)

clip_min_I=np.min([avo_I1, avo_I3])
clip_max_I=np.max([avo_I1, avo_I3])
clip_min_G=np.min([avo_G1, avo_G3])
clip_max_G=np.max([avo_G1, avo_G3])
    
f, ax = plt.subplots(nrows=3,ncols=2,figsize=(12,12))
map0 = ax[0,0].pcolormesh(X, Y, avo_I1, cmap=colr, vmin=clip_min_I, vmax=clip_max_I)
map1 = ax[0,1].pcolormesh(X, Y, avo_G1, cmap=colr, vmin=clip_min_G, vmax=clip_max_G)
map2 = ax[1,0].pcolormesh(X, Y, avo_I3, cmap=colr, vmin=clip_min_I, vmax=clip_max_I)
map3 = ax[1,1].pcolormesh(X, Y, avo_G3, cmap=colr, vmin=clip_min_G, vmax=clip_max_G)
map4 = ax[2,0].pcolormesh(X, Y, avo_I1-avo_I3, cmap='RdGy', vmin=-10, vmax=10)
map5 = ax[2,1].pcolormesh(X, Y, avo_G1-avo_G3, cmap='RdGy', vmin=-10, vmax=10)

ax[0,0].set_title('I (numpy.polyfit)')
ax[0,1].set_title('G (numpy.polyfit)')
ax[1,0].set_title('I (multiple_linregress)')
ax[1,1].set_title('G (multiple_linregress)')
ax[2,0].set_title('I diff')
ax[2,1].set_title('G diff')

plt.colorbar(map0, ax=ax[0,0], shrink=0.5)
plt.colorbar(map1, ax=ax[0,1], shrink=0.5)
plt.colorbar(map2, ax=ax[1,0], shrink=0.5)
plt.colorbar(map3, ax=ax[1,1], shrink=0.5)
plt.colorbar(map4, ax=ax[2,0], shrink=0.5)
plt.colorbar(map5, ax=ax[2,1], shrink=0.5)

avo_I3=avo_I3.reshape(X.shape)
avo_G3=avo_G3.reshape(Y.shape)

clip_max=np.max(hrz_extr[:,3:4])
clip_min=np.min(hrz_extr[:,3:4])

f, ax = plt.subplots(nrows=3,ncols=2,figsize=(12,12))
map0 = ax[0,0].pcolormesh(X, Y, An,     cmap=colr, vmin=clip_min,   vmax=clip_max)
map1 = ax[0,1].pcolormesh(X, Y, Af,     cmap=colr, vmin=clip_min,   vmax=clip_max)
map2 = ax[1,0].pcolormesh(X, Y, I,      cmap=colr, vmin=clip_min_I, vmax=clip_max_I)
map3 = ax[1,1].pcolormesh(X, Y, G,      cmap=colr, vmin=clip_min_G, vmax=clip_max_G)
map4 = ax[2,0].pcolormesh(X, Y, avo_I3, cmap=colr, vmin=clip_min_I, vmax=clip_max_I)
map5 = ax[2,1].pcolormesh(X, Y, avo_G3, cmap=colr, vmin=clip_min_G, vmax=clip_max_G)
ax[0,0].set_title('Near')
ax[0,1].set_title('Far')
ax[1,0].set_title('AVO Intercept (original)')
ax[1,1].set_title('AVO Gradient (original)')
ax[2,0].set_title('AVO Intercept (new)')
ax[2,1].set_title('AVO Gradient (new)')
plt.colorbar(map0, ax=ax[0,0], shrink=0.5)
plt.colorbar(map1, ax=ax[0,1], shrink=0.5)
plt.colorbar(map2, ax=ax[1,0], shrink=0.5)
plt.colorbar(map3, ax=ax[1,1], shrink=0.5)
plt.colorbar(map4, ax=ax[2,0], shrink=0.5)
plt.colorbar(map5, ax=ax[2,1], shrink=0.5)

