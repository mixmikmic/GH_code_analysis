# To be able to run this code you will need Cython installed (sudo pip install Cython)
# and to run the command
# python setup_dtw.py build_ext --inplace
# To compile the Cython library

get_ipython().run_cell_magic('bash', '', 'cd dtwpy\npython setup_dtw.py build_ext --inplace')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import itertools

from dtwpy import dtw_distances, dtw
from dtwpy.plot import plot_alignment, plot_cost_matrix, plot_distances, plot_cost_matrices

get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
np.set_printoptions(precision=3)

metrics = ['sqeuclidean','euclidean','cosine']
dtw_parameters = [
    {},
    {'constraint' : 'sakoe_chiba',  'k' : 0.05},
    {'constraint' : 'sakoe_chiba',  'k' : 0.1},
    {'constraint' : 'sakoe_chiba',  'k' : 0.2},
    {'constraint' : 'slanted_band', 'k' : 0.05},
    {'constraint' : 'slanted_band', 'k' : 0.1},
    {'constraint' : 'slanted_band', 'k' : 0.2},
    {'constraint' : 'itakura'},
]
labels = [
    'Unconstrained',
    'Sakoe Chiba 0.05',
    'Sakoe Chiba 0.1',
    'Sakoe Chiba 0.2',
    'Slanted Band 0.05',
    'Slanted Band 0.1',
    'Slanted Band 0.2',
    'Itakura Parallelogram'
]

lengths = np.array([100,200,500,1000,2000,5000])
y1 = [np.sin(2*np.pi*np.linspace(0,3,N)) for N in lengths]
y2 = [-0.15*np.cos(3*2*np.pi*np.linspace(0,3,N))+np.cos(2*np.pi*np.linspace(0,3,N)) for N in lengths]

u,v = y1[2],y2[2]
plt.plot(u)
plt.plot(v)

dist, cost_matrix, (alig_u, alig_v) = dtw(u,v,dist_only=False)
print(dist)
plt.plot(alig_v, alig_u,color='g')
plt.imshow(cost_matrix, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
plt.figure()

plt.plot(u[alig_u])
plt.plot(v[alig_v])

dist, cost_matrix, (alig_u, alig_v) = dtw(u,v,dist_only=False,constraint='sakoe_chiba',k=0.2)
print(dist)
plt.plot(alig_v, alig_u,color='g')
plt.imshow(cost_matrix, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
plt.figure()

plt.plot(u[alig_u])
plt.plot(v[alig_v])

dist, cost_matrix, (alig_u, alig_v) = dtw(u,v,dist_only=False,constraint='itakura')
print(dist)
plt.plot(alig_v, alig_u,color='g')
plt.imshow(cost_matrix, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
plt.figure()

plt.plot(u[alig_u])
plt.plot(v[alig_v])

try:
    from tqdm import tqdm_notebook
except:
    def tqdm_notebook(x,**kwargs):
        return x
it = 10
times = np.zeros([len(lengths),len(labels)])
for i,kwargs in tqdm_notebook(enumerate(dtw_parameters),desc='constraint',total=len(dtw_parameters)):
    times[:,i] = np.array([timeit.Timer("dtw(y1[%d],y2[%d],**kwargs)" % (i,i),                            setup="from __main__ import dtw,y1,y2,kwargs").timeit(number=it)/it                            for i,_ in tqdm_notebook(enumerate(y1),desc='lengths',leave=False,total=len(y1))])

df_times = pd.DataFrame(times,columns=labels,index=lengths)
df_times.plot.bar(logy=True)
plt.xlabel('Length of signals')
plt.ylabel('Time[s]')

# Some harmonic approximations of Square Wave
long_sigs = [ np.sum(np.array([ 1/j*np.sin(2*np.pi*j*np.linspace(0,1,2500)) for j in range(1,i+1,2) ]),axis=0) for i in range(1,21,2)]

dtw_distances(long_sigs)

dtw_distances(long_sigs,n_jobs=2)

long_sigs = [ np.sum(np.array([ 1/j*np.sin(2*np.pi*j*np.linspace(0,1,2500)) for j in range(1,i+1,2) ]),axis=0) for i in range(1,21,2)]
it = 1
n_procs = [1,2,4,-1]
times = np.zeros([len(n_procs),len(labels)])
for i,kwargs in tqdm_notebook(enumerate(dtw_parameters),total=len(dtw_parameters),desc='constraint'):
    times[:,i] = np.array([timeit.Timer("dtw_distances(long_sigs,n_jobs=%d,**kwargs)" % (n),                            setup="from __main__ import dtw_distances,long_sigs,kwargs").timeit(number=it)/it                            for n in tqdm_notebook(n_procs,desc='n_proc',leave=False)])

df_times = pd.DataFrame(times,columns=labels,index=n_procs[:-1]+['all'])
df_times.plot.bar()
plt.xlabel('number of processors')
plt.ylabel('Time[s]')

sigs = [ np.sum(np.array([ 1/j*np.sin(2*np.pi*j*np.linspace(0,1)) for j in range(1,i+1,2) ]),axis=0) for i in range(1,13,2)]
sigs = [np.sin(2*np.pi*np.linspace(0,1)+1)]+sigs
u,v = sigs[2],sigs[5]

fig = plot_alignment(u,v,normalize=True,constraint='itakura')

fig = plot_cost_matrix(u,v,normalize=True,constraint='sakoe_chiba',k=0.25,axis=True)

fig = plot_distances(sigs,normalize=True,constraint='sakoe_chiba',k=0.25)

fig = plot_cost_matrices(sigs,normalize=True,constraint='sakoe_chiba',k=0.25,vmax=10)

get_ipython().magic('pinfo dtw')



