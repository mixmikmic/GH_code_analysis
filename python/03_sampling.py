get_ipython().run_line_magic('matplotlib', 'inline')
import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')
sns.set_color_codes()

p_grid = np.linspace(0, 1, 100)
prior = np.repeat(1, 100)
likelihood = stats.binom.pmf(k=6, n=9, p=p_grid)
posterior = (likelihood * prior)/(likelihood * prior).sum()

plt.plot(p_grid, posterior)

samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)

sns.distplot(samples)
plt.plot(p_grid, posterior*1e2, color='red')

plt.plot(samples, 'bo', alpha=0.05)

# What is the probability that the coin bias is < 50%
p_lt_5 = np.sum(samples < 0.5)/1E4

# What is the probability that the coin bias is between 0.5 and 0.75
p_bw_050_075 = np.sum((samples > 0.5) & (samples < 0.75))/1E4

# What is the 80th percentile of the coin bias?
p_pct_80 = np.percentile(samples, 80)

# 10th and 90th percentiles of posterior
pct_10_and_90 = np.percentile(samples, (10, 90))
p_lt_5, p_bw_050_075, p_pct_80, pct_10_and_90

# Print results
print('P(bias < 0.5) = %s' % p_lt_5)
print('P(0.5 <= bias <= 0.75) = %s' % p_bw_050_075)
print('π(x) means xth percentile')
print('π(0.8) = %s' % p_pct_80)
print('π(0.1) = %s, π(0.9) = %s' % (pct_10_and_90[0], pct_10_and_90[1]))

p_grid = np.linspace(0, 1, 100)
prior = np.repeat(1, 100)
likelihood = stats.binom.pmf(k=3, n=3, p=p_grid)
posterior = (likelihood * prior)/(likelihood * prior).sum()

samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)

sns.distplot(samples)
plt.plot(p_grid, posterior*1e2, color='red')

pct_25_75 = np.percentile(samples, [25, 75])
hpd = pm.hpd(samples, 0.5)
mean = samples.mean()
median = np.median(samples)
mode = stats.mode(samples)

pct_25_75, hpd, mean, median, mode

d_grid = np.linspace(0, 1, 1000)

results = np.zeros(1000)
# FIXME: Vectorize this
for i, v in enumerate(d_grid):
    results[i] = (posterior * abs(v - p_grid )).sum() 

plt.plot(d_grid, results)
plt.xlabel('Decision')
plt.ylabel('Loss for a particular decision')

d_grid[np.argmin(results)]

from scipy.optimize import minimize
# Using `Nelder-Mead` instead of `BFGS` because BFGS has hard time converging. Why???
d_optimal = minimize(lambda d: (posterior*abs(d-p_grid)).sum(), 0.0, options={'xtol': 1e-2}, method='Nelder-Mead')
d_optimal

p_grid = np.linspace(0, 1, 100)
prior = np.repeat(1, 100)
likelihood = stats.binom.pmf(k=6, n=9, p=p_grid)
posterior = (likelihood * prior)/(likelihood * prior).sum()

samples = np.random.choice(p_grid, p=posterior, size=int(1e4), replace=True)

sns.distplot(samples)
plt.plot(p_grid, posterior*1e2, color='red')

plt.hist(np.random.binomial(9, 0.7, 10000), normed=True)

# FIXME: figure out broadcasting
plt.hist(np.random.binomial(9, p=samples), normed=True)

