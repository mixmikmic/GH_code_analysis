get_ipython().magic('pylab inline')

import numpy as np
x = np.linspace(0, 10, 100)
pylab.plot(x, np.sin(x))

