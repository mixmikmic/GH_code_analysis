get_ipython().magic('matplotlib inline')

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b['q'] = 0.7

b['rpole@primary'] = 1.0

b['rpole@secondary'] = 0.5

b['teff@secondary@component'] = 5000

b['syncpar@primary@component'] = 2

b.add_dataset('rv', times=np.linspace(0,2,201), dataset='dynamicalrvs')  
# TODO: can't set rv_method here because compute options don't exist yet... and that's kind of annoying

b.add_dataset('rv', times=np.linspace(0,2,201), dataset='numericalrvs')

times = b.get_value('times@primary@numericalrvs@dataset')
times = times[times<0.1]
print times

b.add_dataset('mesh', dataset='mesh01', times=times)

b.set_value_all('rv_method@dynamicalrvs@compute', 'dynamical')
b.set_value_all('rv_method@numericalrvs@compute', 'flux-weighted')

print b['rv_method']

b.run_compute(irrad_method='none')

axs, artists = b['dynamicalrvs@model'].plot(component='primary', color='b')
axs, artists = b['dynamicalrvs@model'].plot(component='secondary', color='r')

axs, artists = b['numericalrvs@model'].plot(component='primary', color='b')
axs, artists = b['numericalrvs@model'].plot(component='secondary', color='r')

fig = plt.figure(figsize=(12,12))
axs, artists = b['mesh@model'].plot(time=0.03, facecolor='rvs@numericalrvs', edgecolor=None)

fig = plt.figure(figsize=(12,12))
axs, artists = b['mesh01@model'].plot(time=0.09, facecolor='vzs', edgecolor=None)



