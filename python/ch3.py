import numpy as np
from scipy.stats import binom, hypergeom
import matplotlib.pyplot as plt
from numpy.random import choice
from numpy.random import permutation

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

x, n, p = 3, 5, 0.2
print(binom.pmf(x, n, p))   # PMF
print(binom.pmf(np.arange(5), n, p))    # PMF for multiple values
print(binom.cdf(x, n, p))     # CDF
print(binom.rvs(n, p, size=7))    # Generating Binomial r.v.s

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
X = np.arange(11)
n ,p = 10, 1/2.
axes[0, 0].bar(X, binom.pmf(X, n, p))
n, p = 10, 1/8.
axes[0, 1].bar(X, binom.pmf(X, n, p))
n, p = 100, 0.03
axes[1, 0].bar(X, binom.pmf(X, n, p))
n, p = 9, 4/5.
axes[1, 1].bar(X[:10], binom.pmf(X[:10], n, p))

x, n, w, b = 5, 12, 7, 13
print(hypergeom.pmf(x, w+b, w, n))
print(hypergeom.pmf(np.arange(5), w+b, w, n))
print(hypergeom.cdf(x, w+b, w, n))
print(hypergeom.rvs(w+b, w, n, size=10))

X = np.arange(8)
n, w, b = 12, 7, 13
fig, ax = plt.subplots(2, 2)
ax[0, 0].bar(X, hypergeom.pmf(X, w+b, w, n))
n, w, b = 12, 14, 6
ax[0, 1].bar(X, hypergeom.pmf(X, w+b, w, n))
n, w, b = 12, 3, 17
ax[1, 0].bar(X, hypergeom.pmf(X, w+b, w, n))
n, w, b = 12, 17, 3
ax[1, 1].bar(X, hypergeom.pmf(X, w+b, w, n))

x = [0, 1, 5, 10]
p = [0.25, 0.5, 0.1, 0.15]
print(np.random.choice(x, size=100, replace=True, p=p))

