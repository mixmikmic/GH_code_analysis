import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')


res_path = './res/'
targets_path = './img/'

rtis_list = glob(res_path+'rt_is.*')
targets_0 = glob(targets_path + 'cam1.*_targets')
targets_1 = glob(targets_path + 'cam2.*_targets')
targets_2 = glob(targets_path + 'cam3.*_targets')

# print rtis_list[0]
# print targets_0[0]

print pid,x,y,z,i0,i1,i2,i3
print target0[i0-2]
print target1[i1-2]
print target2[i2-2]

get_ipython().run_line_magic('cat', 'ptc/ptc000.dat')

get_ipython().run_line_magic('cat', 'res/ptc000.dat')

ptv = np.loadtxt('./res/ptc000.dat')
ptc = np.loadtxt('./ptc/ptc000.dat')

fig = plt.figure(figsize=(12,10))
plt.scatter(ptv[:,1],ptv[:,2],color='b')
plt.scatter(ptc[:,1],ptc[:,2],color='r',marker='x')

fig = plt.figure(figsize=(12,10))
plt.scatter(ptv[:,1],ptv[:,3],color='b')
plt.scatter(ptc[:,1],ptc[:,3],color='r',marker='x')

plt.figure()
plt.plot()

