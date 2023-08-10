import numpy as np    # import the whole package numpy 
import matplotlib.pyplot as plt # import the subpacke pyplot
from scipy.stats import norm  # import the function norm

t0 = 20.0
alpha = 0.9
R_0 = 1.3
V = 10
t = np.arange(21, 30, 0.5) # different temperature values ranging from 21 to 30 in steps of 0.5

P = V**2 / (R_0 * (1+alpha*(t-t0)))

plt.figure(1)
plt.plot(t, P)
plt.xlabel("temperature / °C")
plt.ylabel("power / W")
plt.show()

