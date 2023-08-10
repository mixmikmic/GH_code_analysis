get_ipython().magic('matplotlib inline')

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b.add_dataset('mesh', times=[0.75], dataset='mesh01')

b['rpole@primary@component'] = 1.8

b.run_compute(irrad_method='none', distortion_method='roche', model='rochemodel')

b.run_compute(irrad_method='none', distortion_method='rotstar', model='rotstarmodel')

axs, artists = b['rochemodel'].plot()

axs, artists = b['rotstarmodel'].plot()



