get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import IPython
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
from sys import version 
import multiprocessing
from numpy import ceil, mean
import time
import os
print ' Reproducibility conditions for this notebook '.center(90,'-')
print 'Python version:     ' + version
print 'Numpy version:      ' + np.__version__
print 'IPython version:    ' + IPython.__version__
print 'Multiprocessing:    ' + multiprocessing.__version__
print '-'*90

def fibonacci_python(n):
    a, b = 0, 1
    while b < n:
        #print b,
        a, b = b, a + b

get_ipython().magic('timeit fibonacci_python(100000000)')

get_ipython().magic('load_ext cython')

get_ipython().run_cell_magic('cython', '', 'def fibonacci_cython(int n ):\n    cdef int a=0, b=1\n    while b < n:\n        #print b,\n        a, b = b, a + b')

get_ipython().magic('timeit fibonacci_cython(100000000)')

def step():
    return np.sign(np.random.random(1)-.5)

def sim1(n):
    x = np.zeros(n)
    dx = 1./n
    for i in xrange(n-1):
        x[i+1] = x[i] + dx * step()
    return x

n = 10000
get_ipython().magic('timeit sim1(n)')

get_ipython().run_cell_magic('cython', '', 'import numpy as np\ncimport numpy as np\nDTYPE = np.double\nctypedef np.double_t DTYPE_t\nfrom libc.stdlib cimport rand, RAND_MAX\nfrom libc.math cimport round\ncdef double step():\n    return 2 * round(float(rand()) / RAND_MAX) - 1\ndef sim2(int n):\n    cdef int i\n    cdef double dx = 1. / n\n    cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros(n, dtype=DTYPE)\n    for i in range(n - 1):\n        x[i+1] = x[i] + dx * step()\n    return x')

get_ipython().magic('timeit sim2(n)')

scenarios = {'1': n, 
             '2': n, 
             '3': n,
             '4': n,
             '5': n,
             '6': n}
results = {}
print '-' * 85
for num_processes in scenarios:
    N = scenarios[num_processes]
    chunks = [int(ceil(N / int(num_processes)))] * int(num_processes)
    chunks[-1] = int(chunks[-1] - sum(chunks) + N)
    p = multiprocessing.Pool(int(num_processes))
    print 'Number of processors:', num_processes 
    get_ipython().magic('timeit p.map(sim1, chunks)')
    p.close()
    p.join()
    print '-' * 85

scenarios = {'1': n, 
             '2': n, 
             '3': n,
             '4': n,
             '5': n,
             '6': n}
results = {}
print '-' * 85
for num_processes in scenarios:
    N = scenarios[num_processes]
    chunks = [int(ceil(N / int(num_processes)))] * int(num_processes)
    chunks[-1] = int(chunks[-1] - sum(chunks) + N)
    p = multiprocessing.Pool(int(num_processes))
    print 'Number of processors:', num_processes 
    get_ipython().magic('timeit p.map(sim2, chunks)')
    p.close()
    p.join()
    print '-' * 85



