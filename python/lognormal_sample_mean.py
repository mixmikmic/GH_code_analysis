get_ipython().magic('matplotlib inline')


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

v = 1.0
true_mean = np.exp(v/2)  # Lognormal case

N = 20
num_reps = 10000
xbar_outcomes = np.empty(num_reps)

for i in range(num_reps):
    x = norm.rvs(size=N)
    x = np.exp(x)
    xbar_outcomes[i] = x.mean()


fig, ax = plt.subplots(figsize=(10, 8))

ax.hist(xbar_outcomes, bins=88, alpha=0.32, normed=True, label='sample mean')
ax.vlines([true_mean], [0], [1.0], lw=2, label='true mean')

ax.set_xlim(0.5, 4)
ax.set_ylim(0, 1)

ax.legend()

plt.show()



