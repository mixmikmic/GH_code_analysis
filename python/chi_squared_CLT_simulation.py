get_ipython().magic('matplotlib inline')


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

num_replications = 5000
outcomes = np.empty(num_replications)
N = 1000
k = 5     # Degrees of freedom
chi = st.chi2(k)

for i in range(num_replications):
    xvec = chi.rvs(N)
    outcomes[i] = np.sqrt(N / (2 * k)) * (xvec.mean() - k)

xmin, xmax = -4, 4
grid = np.linspace(xmin, xmax, 200)

fig, ax, = plt.subplots(figsize=(10, 8))
ax.hist(outcomes, bins=50, normed=True, alpha=0.4)
ax.plot(grid, st.norm.pdf(grid), 'k-', lw=2, alpha=0.7)
plt.show()




