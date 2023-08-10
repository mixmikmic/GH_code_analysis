get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
import mpld3

mpld3.enable_notebook()

# Turn off matplotlib interactive mode
# plt.ioff()

plt.plot(np.array([3,1,4,1,5]), 'ks-', mec='w', mew=5, ms=20)

n = 50
x, y, z, s, ew = np.random.rand(5, n)
c, ec = np.random.rand(2, n, 4)
area_scale, width_scale = 500, 5

fig, ax = plt.subplots()
sc = ax.scatter(x, y, 
                c=c,
                s=np.square(s)*area_scale,
                edgecolor=ec,
                # The linewidth parameter must be a list, i.e., it cannot be
                # a NumPy array. Without the conversion to a list, the call
                # to scatter will throw an exception since a NumPy array
                # cannot be serialized to JSON.
                linewidth=list(ew*width_scale))
ax.grid()



