import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import nelpy as nel  # recommended import for nelpy
import nelpy.plotting as npl  # recommended import for the nelpy plotting library

get_ipython().run_line_magic('matplotlib', 'inline')

epocharray = nel.EpochArray([[3,4],[5,8],[10,12], [16,20], [22,23]])
data = [3,4,2,5,2]

for epoch in epocharray:
    print(epoch)

for epoch, val in zip(epocharray, data):
    plt.plot([epoch.start, epoch.stop], [val, val], '-o', color='k', markerfacecolor='w', lw=1.5, mew=1.5)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  
    for epoch, val in zip(epocharray, data):
        plt.plot([epoch.start, epoch.stop], [val, val], '-o', color='k', markerfacecolor='w', lw=1.5, mew=1.5)

warnings.simplefilter("ignore") 

ax = npl.plot(epocharray, data, color='0.3')

ax = npl.plot(epocharray, data, color='deepskyblue')

ax = npl.plot(epocharray, data, color='deepskyblue', mec='red')

ax = npl.plot(epocharray, data, color='k', lw=1, marker='d', mew=3, mec='orange', ms=14, linestyle='dashed')

# create dictionary of frequently used parameters:
kws = {'lw': 5, 'ms':0, 'color': 'orange'}  # hide markers, set color to orange, and set linewidth to 5

# pass keyword dictionary to npl.plot():
ax = npl.plot(epocharray, data, **kws)
ax = npl.plot(epocharray, data[::-1], linestyle='dashed', **kws)

