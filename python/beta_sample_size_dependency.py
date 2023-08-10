from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Define function to compute beta
def calc_beta(sample):
    return np.var(sample, ddof = 1)/np.mean(sample)**2

# Define settings
mean_m = 5.07e7   # Mean m from paper
n_iter = 100000    # Number of iterations
# Define artificial sample sizes
n_sample = (range(2,100,1) + range(100, 1000, 20) + range(1000, 2000, 200))

# Loop over sample sizes and compute mean beta over all iterations
beta_list = []
for ns in n_sample:
    tmplist_beta = []
    for ni in range(n_iter):
        sample_beta = np.random.exponential(mean_m, ns)
        tmplist_beta.append(calc_beta(sample_beta))
    beta_list.append(np.mean(tmplist_beta))

# Plot result
fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.plot(n_sample, beta_list, linewidth = 2, c = 'k')
ax.set_xlabel('Sample size')
ax.set_ylabel(r'$\beta$')
ax.set_xscale('log')
ax.set_title(r'Dependency of $\beta$ on sample size')
ax.axhline(1, color='gray', zorder=0.1)
plt.tight_layout()



