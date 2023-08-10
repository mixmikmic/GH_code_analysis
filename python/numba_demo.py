get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np

def qm(x0, n):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
        x[t+1] = 4 * x[t] * (1 - x[t])
    return x

plt.plot(qm(0.1, 100))

n = 10**7

get_ipython().magic('timeit qm(0.1, n)')

from numba import jit

qm_jit = jit(qm)

get_ipython().magic('timeit qm_jit(0.1, n)')

