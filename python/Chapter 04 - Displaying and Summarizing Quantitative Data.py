# %load ./snippets/data-imports
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set_style('whitegrid')

# some data to work with
s = Series(data = [12, 127, 28, 42, 39, 113, 42, 18, 44, 118, 44, 37, 113, 124, 37,
              48, 127, 36, 29, 31, 125, 139, 131, 115, 105, 132, 104, 123, 35,
              113, 122, 42, 117, 119, 58, 109, 23, 105, 63, 27, 44, 105, 99,
              41, 128, 121, 116, 125, 32, 61, 37, 127, 29, 113, 121, 58, 114,
              126, 53, 114, 96, 25, 109, 7, 31, 141, 46, 13, 27, 43, 117, 116,
              27, 7, 68, 40, 31, 115, 124, 42, 128, 52, 71, 118, 117, 38, 27,
              106, 33, 117, 116, 111, 40, 119, 47, 105, 57, 122, 109, 124, 115,
              43, 120, 43, 27, 27, 18, 28, 48, 125, 107, 114, 34, 133, 45, 120,
              30, 127, 31, 116, 146])

# basic histogram - counts
plt.hist(s)
plt.xlabel('value')
plt.ylabel('count')
plt.title('Demo Histogram - Counts')
plt.show()

# relative frequency histogram - %
plt.hist(s, weights=np.zeros_like(s) + 1. / s.size)
plt.xlabel('value')
plt.ylabel('pct')
plt.title('Demo Histogram - Pcts')
plt.show()

# from https://www.rosettacode.org/wiki/Stem-and-leaf_plot#Python
from math import floor
  
def stemplot(values, leafdigits):
    d = []
    interval = int(10**int(leafdigits))
    for data in sorted(values):
        data = int(floor(data))
        stm, lf = divmod(data,interval)
        d.append( (int(stm), int(lf)) )
    stems, leafs = list(zip(*d))
    stemwidth = max(len(str(x)) for x in stems)
    leafwidth = max(len(str(x)) for x in leafs)
    laststem, out = min(stems) - 1, []
    for s,l in d:
        while laststem < s:
            laststem += 1
            out.append('\n%*i |' % ( stemwidth, laststem))
        out.append(' %0*i' % (leafwidth, l))
    out.append('\n\nKey:\n Stem multiplier: %i\n X | Y  =>  %i*X+Y\n'
               % (interval, interval))
    return ''.join(out)

print(stemplot(s, 1.0))

# TODO: find implementation using Matplotlib...

s.median()

s.max() - s.min()

s.quantile(0.75) - s.quantile(0.25)

s.describe()

s.mean()

# sample std dev
s.std()

# population std dev (/n rather than /n-1)
s.std(ddof=0)



