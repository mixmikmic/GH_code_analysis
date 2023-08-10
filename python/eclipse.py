get_ipython().magic('matplotlib inline')

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b.add_dataset('mesh', times=[0.05])

b.run_compute(eclipse_method='native')

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
axs, artists = b.plot(component='primary', facecolor='visibilities', ax=ax)

wcs = b['visible_centroids@primary'].get_value()

ax.plot(wcs[:,0], wcs[:,1], 'b.')
xlim = ax.set_xlim(-0.5,0.25)
ylim = ax.set_ylim(-0.4,0.4)

b.run_compute(eclipse_method='visible_partial')

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
axs, artists = b.plot(component='primary', facecolor='visibilities', ax=ax)

xlim = ax.set_xlim(-0.5,0.25)
ylim = ax.set_ylim(-0.4,0.4)



