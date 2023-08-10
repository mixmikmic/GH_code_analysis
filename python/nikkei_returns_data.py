get_ipython().magic('matplotlib inline')

import pandas as pd

df = pd.read_csv('../data/nikkei.csv', names=['date', 'price'], header=0, na_values='.')
df = df.dropna()
returns = df.price.pct_change()
returns = returns.dropna()

# Compute standardized returns
r = (returns - returns.mean()) / returns.std()

r = r.dropna()

import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from scipy.stats import norm


rmin = r.min()
rmax = r.max()
rgrid = np.linspace(rmin, rmax, 200)

e = qe.ecdf.ECDF(r)

temp = norm.cdf(rgrid) - [e(rg) for rg in rgrid]
T = np.sqrt(len(r)) * np.max(np.abs(temp))
print(T)

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(rgrid, [e(rg) for rg in rgrid], label='empirical distribution')
ax.plot(rgrid, norm.cdf(rgrid), label='standard normal cdf')

ax.legend(loc='upper left')

plt.show()


import numpy as np
import quantecon as qe
from scipy.stats import norm


rmin = r.min()
rmax = r.max()
rgrid = np.linspace(rmin, rmax, 200)

fig, ax = plt.subplots(figsize=(12, 9))

ax.plot(rgrid, norm.pdf(rgrid), label='standard normal pdf', lw=2)

ax.hist(r.values, bins=32, normed=True, label='standardized returns', alpha=0.3)

ax.legend(loc='upper left')

plt.show()



