get_ipython().magic('load_ext Cython')

get_ipython().run_cell_magic('cython', '', "\nimport numpy as np \n\ndef trivial_func(double[:] x):\n    cdef double[:] y = np.zeros(5, dtype='f8')\n    cdef int[:] z = np.ones(5, dtype='i4')\n    return x[0] + y[0] + z[0]")

trivial_func(np.ones(4))

get_ipython().run_cell_magic('cython', '', "\nimport numpy as np \n\ndef typed_memoryiews_do_not_support_broadcasting(double[:] x):\n    cdef double[:] y = np.zeros(5, dtype='f8')\n    return x + y")

get_ipython().run_cell_magic('cython', '', "\nimport numpy as np \n\ndef summing_x_and_y(double[:] x, double[:] y):\n    cdef int i\n    cdef int npts = len(y)\n    cdef double[:] result = np.zeros(npts, dtype='f8')\n    for i in range(npts):\n        result[i] = x[i] + y[i]\n    return result")

summing_x_and_y(np.random.rand(10), np.random.rand(10))

get_ipython().run_cell_magic('cython', '', "\nimport numpy as np \n\ndef summing_x_and_y_array_result(double[:] x, double[:] y):\n    cdef int i\n    cdef int npts = len(y)\n    cdef double[:] result = np.zeros(npts, dtype='f8')\n    for i in range(npts):\n        result[i] = x[i] + y[i]\n    return np.array(result)")

summing_x_and_y_array_result(np.random.rand(10), np.random.rand(10))



