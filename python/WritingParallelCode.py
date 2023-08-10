from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
get_ipython().magic('precision 4')
plt.style.use('ggplot')

get_ipython().magic('load_ext cythonmagic')

from numba import jit, typeof, int32, int64, float32, float64

import random

def pi_python(n):
    s = 0
    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if (x**2 + y**2) < 1:
            s += 1
    return s/n

stats = get_ipython().magic('prun -r -q pi_python(1000000)')

stats.sort_stats('time').print_stats(5);

def pi_numpy(n):
    xs = np.random.uniform(-1, 1, (n,2))
    return 4.0*((xs**2).sum(axis=1).sum() < 1)/n

@jit
def pi_numba(n):
    s = 0
    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            s += 1
    return s/n

get_ipython().run_cell_magic('cython', '-a -lgsl', 'import cython\nimport numpy as np\ncimport numpy as np\nfrom cython_gsl cimport gsl_rng_mt19937, gsl_rng, gsl_rng_alloc, gsl_rng_uniform\n\ncdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)\n\n@cython.cdivision\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef pi_cython(int n):\n    cdef int[:] s = np.zeros(n, dtype=np.int32)\n    cdef int i = 0\n    cdef double x, y\n    for i in range(n):\n        x = gsl_rng_uniform(r)*2 - 1\n        y = gsl_rng_uniform(r)*2 - 1\n        s[i] = x**2 + y**2 < 1\n    cdef int hits = 0\n    for i in range(n):\n        hits += s[i]\n    return 4.0*hits/n')

n = int(1e5)
get_ipython().magic('timeit pi_python(n)')
get_ipython().magic('timeit pi_numba(n)')
get_ipython().magic('timeit pi_numpy(n)')
get_ipython().magic('timeit pi_cython(n)')

import multiprocessing

num_procs = multiprocessing.cpu_count()
num_procs

def pi_multiprocessing(n):
    """Split a job of length n into num_procs pieces."""
    import multiprocessing
    m = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(m)
    results = pool.map(pi_cython, [n/m]*m)
    pool.close()
    return np.mean(results)

n = int(1e5)
get_ipython().magic('timeit pi_cython(n)')
get_ipython().magic('timeit pi_multiprocessing(n)')

n = int(1e7)
get_ipython().magic('timeit pi_numpy(n)')
get_ipython().magic('timeit pi_multiprocessing(n)')

from multiprocessing import Pool, Value, Array, Lock, current_process

n = 4
val = Value('i')
arr = Array('i', n)

val.value = 0
for i in range(n):
    arr[i] = 0

def count1(i):
    "Everyone competes to write to val."""
    val.value += 1
    
def count2(i):
    """Each process has its own slot in arr to write to."""
    ix = current_process().pid % n
    arr[ix] += 1
    
pool = Pool(n)
pool.map(count1, range(1000))
pool.map(count2, range(1000))

pool.close()
print val.value
print sum(arr)

from IPython.parallel import Client, interactive

rc = Client()
print rc.ids
dv = rc[:]

get_ipython().magic('px import numpy as np')

for i, r in enumerate(rc):
    r.execute('np.random.seed(123)')

get_ipython().run_cell_magic('px', '', '\nnp.random.random(3)')

dv.scatter('seed', [1,1,2,2], block=True)

dv['seed']

get_ipython().run_cell_magic('px', '', '\nnp.random.seed(seed)\nnp.random.random(3)')

for i, r in enumerate(rc):
    r.execute('np.random.seed(%d)' % i)

get_ipython().run_cell_magic('px', '', '\nnp.random.random(3)')

get_ipython().run_cell_magic('px', '', '\nx = np.random.random(3)')

dv['x']

dv.gather('x', block=True)

get_ipython().run_cell_magic('px', '', 'n = 1e7\nx = np.random.uniform(-1, 1, (n, 2))\nn = (x[:, 0]**2 + x[:,1]**2 < 1).sum()')

get_ipython().magic('precision 8')
ns = dv['n']
4*np.sum(ns)/(1e7*len(rc))

dv.scatter('s', np.arange(16), block=False)

dv['s']

dv.gather('s')

dv.gather('s').get()

ar = dv.map_async(lambda x: x+1, range(10))
ar.ready()

ar.ready()

ar.get()

lv = rc.load_balanced_view()

def wait(n):
    import time
    time.sleep(n)
    return n

dv['wait'] = wait

intervals = [5,1,1,1,1,1,1,1,1,1,1,1,1,5,5,5]

get_ipython().run_cell_magic('time', '', '\nar = dv.map(wait, intervals)\nar.get()')

get_ipython().run_cell_magic('time', '', '\nar = lv.map(wait, intervals, balanced=True)\nar.get()')

get_ipython().run_cell_magic('px', '', 'def python_loop(xs):\n    s = 0.0\n    for i in range(len(xs)):\n        s += xs[i]\n    return s')

get_ipython().run_cell_magic('px', '', '%load_ext cythonmagic')

get_ipython().run_cell_magic('px', '', '%%cython\n\ndef cython_loop(double[::1] xs):\n    n = xs.shape[0]\n    cdef int i\n    cdef double s = 0.0\n    for i in range(n):\n        s += xs[i]\n    return s')

get_ipython().run_cell_magic('time', '', '%%px\nxs = np.random.random(1e7)\ns = python_loop(xs)')

dv['s']

get_ipython().run_cell_magic('time', '', '%%px\nxs = np.random.random(1e7)\ns = cython_loop(xs)')

dv['s']

get_ipython().magic('load_ext version_information')

get_ipython().magic('version_information numba, multiprocessing, cython')



