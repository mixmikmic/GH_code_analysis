import numpy as np
from math import cos, log

# A loop within a loop as primitive Python code.  

def f_py(I, J): 
    res = 0
    for i in range(I):
        for j in range (J):
            res += int(cos(log(1)))
    return res

# Set both loops at 10000
# Given the calculation in the loops, we are calculating 10000 * 10000

I, J = 10000, 10000
get_ipython().magic('time res = f_py(I, J)')

print(res)

# Now use the more efficient Numpy arrays

def f_np(I, J):
    a = np.ones((I, J), dtype=np.float64)
    return int(np.sum(np.cos(np.log(a))))

get_ipython().magic('time res = f_np(I, J)')

print(res)

# Import Numba, which considerably speeds up looping.  
# See http://numba.pydata.org/ for an explanation of how it does this.

import numba as nb
f_py_nb = nb.jit(f_py)
f_np_nb = nb.jit(f_np)

get_ipython().magic('time f_py_nb(I, J)')

get_ipython().magic('time f_np_nb(I, J)')

