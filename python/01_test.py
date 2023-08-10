import numpy as np; print(np.__version__)

import scipy as sp; print(sp.__version__)

import pandas as pd; print(pd.__version__)

import xarray as xr; print(xr.__version__)

import matplotlib as mpl; print(mpl.__version__)

from mpl_toolkits import basemap; print(basemap.__version__)

import cartopy; print(cartopy.__version__)

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.plot(np.sin(np.linspace(-np.pi, np.pi, 100)))



