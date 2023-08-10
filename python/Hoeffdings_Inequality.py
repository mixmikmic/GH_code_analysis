# style the notebook
from IPython.core.display import HTML
import urllib.request
response = urllib.request.urlopen('http://bit.ly/1LC7EI7')
HTML(response.read().decode("utf-8"))

import math

def hoeffding(epsilon, N):
    return 2*math.exp(-2 * N * (epsilon**2))

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
eps = 0.1
N = 1000

eps_range = np.arange(1.e-3, .1, .001)
hs = []
for eps in eps_range:
    hs.append(hoeffding(eps, N))
plt.plot(eps_range, hs);    

ns = np.arange(1, 1000)
hs = []
for N in ns:
    hs.append(hoeffding(0.05, N))
plt.plot(ns, hs);

