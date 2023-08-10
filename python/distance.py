get_ipython().magic('matplotlib inline')

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

print b.get_parameter(qualifier='distance', context='system')

print b.get_parameter(qualifier='t0', context='system')

b.add_dataset('orb', times=np.linspace(0,3,101), dataset='orb01')

b.set_value('distance', 10.0)

b.run_compute(model='dist10')

b.set_value('distance', 20.0)

b.run_compute(model='dist20')

fig = plt.figure()
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

axs, artists = b['orb01'].plot(model='dist10', ax=ax1)
axs, artists = b['orb01'].plot(model='dist20', ax=ax2)

b.add_dataset('lc', times=np.linspace(0,3,101), dataset='lc01')

b.set_value_all('ld_func', 'logarithmic')
b.set_value_all('ld_coeffs', [0.,0.])

b.set_value('distance', 10.0)

b.run_compute(model='dist10')

b.set_value('distance', 20.0)

b.run_compute(model='dist20')

fig = plt.figure()
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

axs, artists = b['lc01'].plot(model='dist10', ax=ax1)
axs, artists = b['lc01'].plot(model='dist20', ax=ax2)

b.add_dataset('mesh', times=[0], dataset='mesh01')

b.set_value('distance', 10)

b.run_compute(model='dist10')

b.set_value('distance', 20)

b.run_compute(model='dist20')

print "dist10 abs_intensities: ", b.get_value(qualifier='abs_intensities', component='primary', dataset='lc01', model='dist10').mean()
print "dist20 abs_intensities: ", b.get_value(qualifier='abs_intensities', component='primary', dataset='lc01', model='dist20').mean()

print "dist10 intensities: ", b.get_value(qualifier='intensities', component='primary', dataset='lc01', model='dist10').mean()
print "dist20 intensities: ", b.get_value(qualifier='intensities', component='primary', dataset='lc01', model='dist20').mean()

