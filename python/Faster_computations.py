import numpy as np
print "export CFLAGS=\"-I",np.__path__[0]+'/core/include/ $CFLAGS\"'

import numpy as np
from numba import jit, autojit
get_ipython().magic('load_ext Cython')


times=np.arange(0,70,0.01)
print "The size of the time array:", times.size

freq = np.arange(0.1,6.0,0.1)*1.7763123
freq[-20:] = freq[:20]*0.744
amp  = 0.05/(freq**2)+0.01
phi  = freq

def fourier_sum_naive(times, freq, amp, phi):
    mags = np.zeros_like(times)
    for i in xrange(times.size):
        for j in xrange(freq.size):
            mags[i] += amp[j] * np.sin( 2 * np.pi * freq[j] * times[i] + phi[j] )
            
    return mags
    
def fourier_sum_numpy(times, freq, amp, phi):
    return np.sum(amp.T.reshape(-1,1) * np.sin( 2 * np.pi * freq.T.reshape(-1,1) * times.reshape(1,-1) + phi.T.reshape(-1,1)), axis=0)

fourier_sum_naive_numba = autojit(fourier_sum_naive)
fourier_sum_numpy_numba = autojit(fourier_sum_numpy)

#@jit
#def fourier_sum_naive_numba(times, freq, amp, phi):
#    mags = np.zeros_like(times)
#    for i in xrange(times.size):
#        for j in xrange(freq.size):
#            mags[i] += amp[j] * np.sin( 2 * np.pi * freq[j] * times[i] + phi[j] )
#            
#    return mags

#@jit()
#def fourier_sum_numpy_numba(times, freq, amp, phi):
#    return np.sum(amp.T.reshape(-1,1) * np.sin( 2 * np.pi * freq.T.reshape(-1,1) * times.reshape(1,-1) + phi.T.reshape(-1,1)), axis=0)

get_ipython().run_cell_magic('cython', '-a', '\ncimport cython\nimport numpy as np\nfrom libc.math cimport sin, M_PI\n\ndef fourier_sum_cython(times, freq, amp, phi, temp):\n    return np.asarray(fourier_sum_purec(times, freq, amp, phi, temp))\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ncdef fourier_sum_purec(double[:] times, double[:] freq, double[:] amp, double[:] phi, double[:] temp):\n    cdef int i, j, irange, jrange\n    irange=len(times)\n    jrange=len(freq)\n    for i in xrange(irange):\n        temp[i]=0\n        for j in xrange(jrange):\n            temp[i] += amp[j] * sin( 2 * M_PI * freq[j] * times[i] + phi[j] )\n    return temp')

get_ipython().run_cell_magic('cython', '--compile-args=-fopenmp --link-args=-fopenmp --force -a', "\ncimport cython\ncimport openmp\nimport numpy as np\nfrom libc.math cimport sin, M_PI\nfrom cython.parallel import parallel, prange\n\ndef fourier_sum_cython_omp(times, freq, amp, phi, temp):\n    return np.asarray(fourier_sum_purec_omp(times, freq, amp, phi, temp))\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ncdef fourier_sum_purec_omp(double[:] times, double[:] freq, double[:] amp, double[:] phi, double[:] temp):\n    cdef int i, j, irange, jrange\n    irange=len(times)\n    jrange=len(freq)\n    #print openmp.omp_get_num_procs()\n    with nogil, parallel(num_threads=4):\n        for i in prange(irange, schedule='dynamic', chunksize=10):\n            temp[i]=0\n            for j in xrange(jrange):\n                temp[i] += amp[j] * sin( 2 * M_PI * freq[j] * times[i] + phi[j] )\n    return temp   ")

temp=np.zeros_like(times)

amps_naive      = fourier_sum_naive(times, freq, amp, phi)
amps_numpy      = fourier_sum_numpy(times, freq, amp, phi)
amps_numba1     = fourier_sum_naive_numba(times, freq, amp, phi)
amps_numba2     = fourier_sum_numpy_numba(times, freq, amp, phi)
amps_cython     = fourier_sum_cython(times, freq, amp, phi, temp)
amps_cython_omp = fourier_sum_cython_omp(times, freq, amp, phi, temp)

get_ipython().magic('timeit -n 5  -r 5  fourier_sum_naive(times, freq, amp, phi)')
get_ipython().magic('timeit -n 10 -r 10 fourier_sum_numpy(times, freq, amp, phi)')
get_ipython().magic('timeit -n 10 -r 10 fourier_sum_naive_numba(times, freq, amp, phi)')
get_ipython().magic('timeit -n 10 -r 10 fourier_sum_numpy_numba(times, freq, amp, phi)')
get_ipython().magic('timeit -n 10 -r 10 fourier_sum_cython(times, freq, amp, phi, temp)')
get_ipython().magic('timeit -n 10 -r 10 fourier_sum_cython_omp(times, freq, amp, phi, temp)')


import matplotlib.pylab as plt

print amps_numpy-amps_cython

fig=plt.figure()
fig.set_size_inches(16,12)

plt.plot(times,amps_naive         ,'-', lw=2.0)
#plt.plot(times,amps_numpy      - 2,'-', lw=2.0)
plt.plot(times,amps_numba1     - 4,'-', lw=2.0)
#plt.plot(times,amps_numba2     - 6,'-', lw=2.0)
plt.plot(times,amps_cython     - 8,'-', lw=2.0)
#plt.plot(times,amps_cython_omp -10,'-', lw=2.0)

plt.show()



