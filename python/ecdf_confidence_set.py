get_ipython().magic('matplotlib inline')


import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kolmogi
from scipy.stats import t

samp_size = 1000
grid_size = 200
df = 10

xgrid = np.linspace(-3, 3, grid_size)

def ecdf(s, X):
    return(np.mean(X <= s))  

X = t.rvs(df, size=samp_size) 
Y = np.empty(grid_size)

for i in range(grid_size):
    Y[i] = ecdf(xgrid[i], X)

Y_upper = Y + 1.36 / np.sqrt(samp_size)
Y_lower = Y - 1.36 / np.sqrt(samp_size)

fig, ax = plt.subplots(figsize=(12, 8))

ax.fill_between(xgrid, Y_lower, Y_upper, color="blue", alpha=0.4)

ax.plot(xgrid, t.cdf(xgrid, df), 'k-', lw=2, alpha=0.8, label='true $F$')

ax.set_ylim(0, 1)
ax.set_title("sample size = {}".format(samp_size))

ax.legend(loc='upper left', frameon=False)

plt.show()



