get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

get_ipython().magic('load_ext Cython')

get_ipython().run_cell_magic('file', 'mathlib.pxd', 'cdef int max(int a, int b)\ncdef int min(int a, int b)')

get_ipython().run_cell_magic('file', 'mathlib.pyx', 'cdef int max(int a, int b): \n    return a if a > b else b \n\ncdef int min(int a, int b): \n    return a if a < b else b')

get_ipython().run_cell_magic('file', 'setup.py', "from distutils.core import setup \nfrom Cython.Build import cythonize \n\nsetup(name =' Hello', \n      ext_modules = cythonize('mathlib.pyx'))")

get_ipython().system(' python setup.py -q build_ext --inplace')

get_ipython().run_cell_magic('cython', '', 'from mathlib cimport max\n\ndef chebyshev(int x1, int y1, int x2, int y2): \n    return max(abs( x1 - x2), abs( y1 - y2))\n\nprint(chebyshev(1,2,3,4))')

get_ipython().run_cell_magic('file', 'rmath.pyx', '#cython: boundscheck=False\n#cython: wraparound=False\ncimport cython_rmath as r\nimport numpy as np\ncimport numpy as np\n\ncpdef rnorm(int n, double mu=0.0, double sigma=1.0):\n    cdef int i\n    cdef double[:] xs = np.zeros(n)\n    for i in range(n):\n        xs[i] = r.rnorm(mu, sigma)\n    return np.array(xs)')

get_ipython().run_cell_magic('file', 'setup.py', 'from distutils.core import setup\nfrom Cython.Distutils import Extension\nfrom Cython.Distutils import build_ext\nimport cython_rmath\n\next = Extension("rmath",\n                ["rmath.pyx"],\n                define_macros=[(\'MATHLIB_STANDALONE\', \'1\')],\n                include_dirs = [\'.\', \'/Users/cliburn/anaconda/lib/python2.7/site-packages/Cython/Includes\',\n                                \'/Users/cliburn/anaconda/lib/python2.7/site-packages/numpy/core/include\'],\n                library_dirs = [\'.\'],\n                libraries = [\'Rmath\', \'m\',]\n               )\n\nsetup(cmdclass = {\'build_ext\': build_ext}, \n                  ext_modules = [ext])')

get_ipython().system(' python setup.py --quiet build_ext --inplace')

import rmath

rmath.rnorm(10)



