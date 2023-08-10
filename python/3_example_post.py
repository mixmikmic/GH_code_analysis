get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pylab as plt

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

plt.plot(x,y);

