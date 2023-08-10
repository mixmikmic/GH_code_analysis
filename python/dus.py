# This is the first cell with code: set up the Python environment
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function, division
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import binom
import pandas as pd
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML

def weightedRandomSample(n, weights):
    '''
       Weighted random sample of size n drawn with replacement.
       Returns indices of the selected items, and the raw uniform values used to 
       select them.
    '''
    if any(weights < 0):
        print('negative weight in weightedRandomSample')
        return float('NaN')
    else:
        wc = np.cumsum(weights, dtype=float)/np.sum(weights, dtype=float)  # ensure weights sum to 1
        theSam = np.random.random_sample((n,))
        return np.array(wc).searchsorted(theSam), theSam

# illustrate the random sampling code
vals = 10
n = 100
w = np.arange(vals)+1  # linearly increasing weights
wrs, raw = weightedRandomSample(n, w)
print(np.sort(wrs))
fig, ax = plt.subplots(nrows=1, ncols=1)
bins = np.arange(np.min(wrs)-0.5, np.max(wrs)+0.5, 1)
ax.hist(wrs, bins=bins)



