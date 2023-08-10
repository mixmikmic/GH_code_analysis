get_ipython().magic('matplotlib inline')
import scipy
import numpy
import matplotlib.pyplot as plt
from utils import plot_planes_at

# FM plane
strike = 78
dip = 20
# Cross-section plane
strikecs = 110
dipcs = 89

fig = plt.figure()
plot_planes_at(0,0, [strike], [dip], strikecs, dipcs)
plt.grid()



