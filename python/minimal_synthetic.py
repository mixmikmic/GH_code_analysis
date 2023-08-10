get_ipython().magic('matplotlib inline')

import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b.add_dataset('lc', times=np.linspace(0,1,201), dataset='mylc')

b.run_compute(irrad_method='none')

axs, artists = b['mylc@model'].plot()

axs, artists = b['mylc@model'].plot(x='phases')

