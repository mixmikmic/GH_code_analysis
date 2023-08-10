import Cython
print("Cython", Cython.__version__)

get_ipython().magic('load_ext cython')

get_ipython().run_cell_magic('cython', '', "print('hi')")

get_ipython().run_cell_magic('cython', '', '\ndef sum(double[:] arr, int size):\n    cdef int i\n    cdef double s = 0\n    for i in range(size):\n        s += arr[i]\n    return s')

import numpy as np
arr = np.array([1.0, 3.0, 34.0])
assert sum(arr, len(arr)) == 38

