get_ipython().magic('matplotlib inline')

import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, Polygon

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# The current version of NumPy available from conda is issuing a warning 
# message that some behavior will change in the future when used with the 
# current version of matplotlib available from conda. This cell just keeps
# that warning from being displayed.
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

class HeatMapper(plt.cm.ScalarMappable):
    """A callable that maps cold colors to low values, and hot to high.
    """
    def __init__(self, data=None):
        norm = mpl.colors.Normalize(vmin=min(data), vmax=max(data))
        cmap = plt.cm.hot_r
        super(HeatMapper, self).__init__(norm, cmap)
        
    def __call__(self, value):
        return self.to_rgba(value)

