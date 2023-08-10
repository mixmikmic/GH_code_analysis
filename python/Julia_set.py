import numpy as np
import pylab as plt
from matplotlib import cm
from itertools import product
from time import perf_counter
import numexpr as ne
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'notebook')

get_ipython().run_line_magic('load_ext', 'line_profiler')

def julia_iteration(z, c, maxiter=256):
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z**2 + c
    return n

def julia_set(w, h, c, maxiter=256):
    m = np.empty((h, w), dtype=np.uint8)
    for j, i in product(range(h), range(w)):
        z = (i-w/2)/(h/2) + (j-h/2)/(h/2)*1j
        m[j,i] = julia_iteration(z, c, maxiter)

    return m

Image.fromarray(julia_set(800, 600, -0.7+0.3*1j))

get_ipython().run_line_magic('timeit', 'julia_set(2000, 1200, -0.7+0.3*1j)')

get_ipython().run_line_magic('lprun', '-f julia_set julia_set(2000, 1200, -0.7+0.3j, 1)')

def julia_set_block(w, h, c, maxiter=256):
    i0 = (np.arange(w)-w/2)/(h/2)
    j0 = (np.arange(h)-h/2)/(h/2)*1j
    z = j0.reshape(h,1) + i0.reshape(w)
    m = np.zeros((h, w), dtype=np.uint8)
    
    for n in range(maxiter):
        limit = np.abs(z) < 2
        mask = np.where(limit)
        zmasked = z[mask]
        z[mask] = ne.evaluate('zmasked**2 + c')
        m[mask] = n
    
    return m

get_ipython().run_line_magic('timeit', 'julia_set_block(2000, 1200, -0.7 + 0.3*1j)')

get_ipython().run_line_magic('lprun', '-f julia_set_block julia_set_block(2000, 1200, -0.7+0.3j, 1)')

def julia_set_block_fast(w, h, c, maxiter=256):
    i0 = (np.arange(w)-w/2)/(h/2)
    j0 = (np.arange(h)-h/2)/(h/2)*1j
    z = j0.reshape(h,1) + i0.reshape(w)
    m = np.zeros((h, w), dtype=np.uint8)
    
    for n in range(maxiter):
        limit = ne.evaluate('z.real**2 + z.imag**2 < 2')
        mask = np.where(limit)
        zmasked = z[mask]
        z[mask] = ne.evaluate('zmasked**2 + c')
        m[mask] = n
    
    return m

get_ipython().run_line_magic('lprun', '-f julia_set_block_fast julia_set_block_fast(2000, 1200, -0.7+0.3j, 1)')

get_ipython().run_line_magic('timeit', 'julia_set_block_fast(2000, 1200, -0.7 + 0.3*1j)')

Image.fromarray(julia_set_block_fast(2000, 1200, -0.7 + 0.3*1j))

get_ipython().run_line_magic('load_ext', 'Cython')

get_ipython().run_cell_magic('cython', '', '\nimport cython\nimport numpy as np\ncimport numpy as np\n#from std_complex cimport cabs\n\ncdef np.uint8_t julia_iteration(double complex z, double complex c, int maxiter=256) nogil:\n    cdef np.uint8_t n\n    for n in range(maxiter):\n        if z.real**2 + z.imag**2 > 2.0:  #abs(z)\n            return n\n        z = z**2 + c\n    return n\n\n\n@cython.boundscheck(False)\ndef julia_set_cython(int w, int h, double complex c, int maxiter=256):\n    cdef int i, j, n\n    cdef double complex z\n    marr = np.empty((h, w), dtype=np.uint8)\n    cdef np.uint8_t [:,:] m = marr\n    \n    for j in range(h):\n        for i in range(w):\n            z = (i-w/2)/(h/2) + (j-h/2)/(h/2)*1j\n            m[j,i] = julia_iteration(z, c, maxiter)\n                \n    return np.asarray(m)')

m = julia_set_cython(2000, 1200, -0.7 + 0.3*1j, 256)

Image.fromarray(m)

get_ipython().run_line_magic('timeit', 'julia_set_cython(2000, 1200, -0.7 + 0.3*1j, 256)')

import numba

@numba.jit(numba.uint8[:,:](numba.int64, numba.int64, numba.complex128, numba.int64))
def julia_set_numba_fast(w, h, c, maxiter=256):
    m = np.zeros((h, w), dtype=np.uint8)
    creal = c.real
    cimag = c.imag
    
    for j in range(h):
        for i in range(w):
            z = (i-w/2)/(h/2) + (j-h/2)/(h/2)*1j
            
            for n in range(maxiter):
                if z.real**2 + z.imag**2 > 2:
                    m[j,i] = n
                    break
                else:
                    z = z**2 + c
    return m

m = julia_set_numba_fast(2000, 1200, -0.7 + 0.3*1j, 256)

Image.fromarray(m)

get_ipython().run_line_magic('timeit', 'julia_set(2000, 1200, -0.7 + 0.3*1j, 256)')
get_ipython().run_line_magic('timeit', 'julia_set_block_fast(2000, 1200, -0.7 + 0.3*1j, 256)')
get_ipython().run_line_magic('timeit', 'julia_set_numba_fast(2000, 1200, -0.7 + 0.3*1j, 256)')
get_ipython().run_line_magic('timeit', 'julia_set_cython(2000, 1200, -0.7 + 0.3*1j, 256)')

15/0.2



