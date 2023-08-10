# Shannon entropy of a binary random variable

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

p = np.linspace(0.001, 0.999, 100)
entropy = -(1 - p)*np.log(1 - p) - p*np.log(p)

plt.plot(p,entropy)
plt.xlabel("p")
plt.ylabel("entropy")



