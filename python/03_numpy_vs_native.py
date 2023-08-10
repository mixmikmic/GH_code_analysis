import numpy as np
from numpy import shape

get_ipython().magic('time')
L = 1000000
a=np.random.uniform(low=-1,high=1,size=L)
b=np.random.uniform(low=-1,high=1,size=L)

shape(a),shape(b)

a[:5],b[:5]

def python_dot(a,b):
    sum=0
    for i in xrange(len(a)):
        sum+=a[i]*b[i]
    return sum
    

get_ipython().magic('time python_dot(a,b)')

get_ipython().magic('time np.dot(a,b)')

get_ipython().magic('load_ext Cython')

get_ipython().run_cell_magic('cython', '', 'cimport numpy as np  # makes numpy available to cython\n# The following line defines a and b as numpy arrays, cython knows how to deal with those.\ndef cython_dot(np.ndarray[np.float64_t, ndim=1] a,\n                np.ndarray[np.float64_t, ndim=1] b):\n    cdef double sum\n    cdef long i\n    sum=0\n    for i in xrange(a.shape[0]):\n        sum=sum+a[i]*b[i]\n    return sum')

get_ipython().run_cell_magic('time', '', 'cython_dot(a,b)')





