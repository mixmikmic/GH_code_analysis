from __future__ import absolute_import, division, print_function



# uncomment the bottom line in this cell, change the final line of 
# the loaded script to `mpld3.display()` (instead of show).

# %load http://mpld3.github.io/_downloads/linked_brush.py



















get_ipython().magic('matplotlib inline')

import mpld3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context('poster')
# sns.set_style('whitegrid') 
sns.set_style('darkgrid') 
plt.rcParams['figure.figsize'] = 12, 8  # plotsize 

def sinplot(flip=1, ax=None):
    """Demo plot from seaborn."""
    x = np.linspace(0, 15, 500)
    for i in range(1, 7):
        ax.plot(x, np.sin(-1.60 + x + i * .5) * (7 - i) * flip, label=str(i))

# mpld3.enable_notebook()

fig, ax = plt.subplots(figsize=(12, 8))
sinplot(ax=ax)
ax.set_ylabel("y-label")
ax.set_xlabel("x-label")
fig.tight_layout()

mpld3.disable_notebook()



