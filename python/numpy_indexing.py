get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
#%config InlineBackend.figure_format = 'svg'
#%config InlineBackend.figure_format = 'pdf'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import tensorflow as tf

# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 18
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

a = np.reshape(np.arange(6), [3, 2])
b = np.reshape(np.arange(6, 12), [3, 2])

a

b

np.concatenate((a, b), axis=1)

a1 = a[:, np.newaxis]
a1

b1 = b[:, np.newaxis]
b1

np.concatenate((a1, b1), axis=2)

a0 = a[np.newaxis, :]
a0

b0 = b[np.newaxis, :]
b0

ab0 = np.concatenate((a0, b0), axis=0)
ab0

T = np.transpose(ab0, (1, 2, 0))
T

T.reshape(3, -1)

A = np.arange(24).reshape(4, 3, 2)

np.sum(A, 1)







