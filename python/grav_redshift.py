get_ipython().magic('matplotlib inline')

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b.add_dataset('rv', times=np.linspace(0,1,101), dataset='rv01')

b.set_value_all('ld_func', 'logarithmic')
b.set_value_all('ld_coeffs', [0.0, 0.0])
b.set_value_all('atm', 'blackbody')

print b['value@rpole@primary@component'], b['value@rpole@secondary@component']

b.run_compute(rv_method='flux-weighted', rv_grav=True, irrad_method='none', model='defaultradii_true')

b['rpole@primary'] = 0.4
b['rpole@secondary'] = 0.4

b.run_compute(rv_method='flux-weighted', rv_grav=True, irrad_method='none', model='smallradii_true')

b.run_compute(rv_method='flux-weighted', rv_grav=False, irrad_method='none', model='smallradii_false')

fig = plt.figure(figsize=(10,6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

axs, artists = b['rv01@defaultradii_true'].plot(ax=ax1, ylim=(-150,150))
axs, artists = b['rv01@smallradii_true'].plot(ax=ax2, ylim=(-150,150))

fig = plt.figure(figsize=(10,6))
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

axs, artists = b['rv01@smallradii_true'].plot(ax=ax1, ylim=(-150,150))
axs, artists = b['rv01@smallradii_false'].plot(ax=ax2, ylim=(-150,150))

print b['rvs@rv01@primary@defaultradii_true'].get_value().min()
print b['rvs@rv01@primary@smallradii_true'].get_value().min()
print b['rvs@rv01@primary@smallradii_false'].get_value().min()

print b['rvs@rv01@primary@defaultradii_true'].get_value().max()
print b['rvs@rv01@primary@smallradii_true'].get_value().max()
print b['rvs@rv01@primary@smallradii_false'].get_value().max()



