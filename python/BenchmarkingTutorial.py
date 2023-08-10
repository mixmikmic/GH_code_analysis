from __future__ import absolute_import, division, print_function # Py2/3 compat

get_ipython().run_cell_magic('time', '', 'import numpy as np\nlength = 100000\nnp.zeros(length) / np.ones(length)')

from math import sin as msin
from numpy import sin as npsin
from scipy import sin as spsin

from math import pi

angle = pi - 0.1

print("math:")
get_ipython().magic('timeit msin(angle)')
print("numpy:")
get_ipython().magic('timeit npsin(angle)')
print("scipy:")
get_ipython().magic('timeit spsin(angle)')

get_ipython().run_cell_magic('prun', '', '\ndef fast_func():\n    return 1\n\ndef slow_func():\n    for i in range(100000):\n        i**2\n\nfast_func()\nslow_func()')



