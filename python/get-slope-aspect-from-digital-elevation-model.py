import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import elevation
from pydem.dem_processing import DEMProcessor
from matplotlib.colors import ListedColormap
from matplotlib import colors

get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

get_ipython().system('eio clip -o Shasta-30m-DEM.tif --bounds -122.4 41.2 -122.1 41.5 ')

# needs to match above command
filename = 'Shasta-30m-DEM.tif'

# instantiate a processor object
processor = DEMProcessor(filename)

# get magnitude of slope and aspect
mag, aspect = processor.calc_slopes_directions()

mag

aspect

# use a log scale to normalize the scale a bit
plt.imshow(np.log(mag), cmap="magma")

plt.imshow(aspect, cmap = "viridis")

