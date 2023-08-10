get_ipython().magic('load_ext Cython')

get_ipython().run_cell_magic('cython', '', "\nimport numpy as np \n\ndef slow_cython(double[:] arr1, double[:] arr2):\n    \n    cdef int i\n    cdef int npts = len(arr1)\n    cdef double[:] result = np.zeros(npts, dtype='f8')\n    \n    for i in range(npts):\n        result[i] = np.sqrt(arr1[i]) + np.exp(arr2[i])\n    \n    return np.array(result)\n        ")

npts = int(1e6)
x = np.random.rand(npts)
y = np.random.rand(npts)

get_ipython().magic('timeit slow_cython(x, y)')

get_ipython().run_cell_magic('cython', '', "\nimport numpy as np \nfrom libc.math cimport sqrt as c_sqrt\nfrom libc.math cimport exp as c_exp\n\ndef fast_cython(double[:] arr1, double[:] arr2):\n    \n    cdef int i\n    cdef int npts = len(arr1)\n    cdef double[:] result = np.zeros(npts, dtype='f8')\n    \n    for i in range(npts):\n        result[i] = c_sqrt(arr1[i]) + c_exp(arr2[i])\n    \n    return np.array(result)")

get_ipython().magic('timeit fast_cython(x, y)')

arr = np.arange(5)
for x in arr:
    print(x)

npts = len(arr)
for i in range(npts):
    print(arr[i])



