import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os # os is a library that allows us to execute certain file and directory operations

cwd = os.getcwd()

#If on MAC, this will likely work
datadir = '/'.join(cwd.split('/')[0:-1]) + '/data/'
#If on window's machine, explicitly put in data dir
#datadir = 

table = pd.read_table(datadir + 'nerve.dat', sep = '\t', header=None)

# A little processing:
nerve_firings = pd.Series()

for i in range(6):
    nerve_firings = nerve_firings.append(table[i].dropna(axis = 0), ignore_index = True)

nerve_firings.describe()

fig, axes = plt.subplots()

nerve_firings.plot(kind = 'hist', normed = True, cumulative = True)
plt.show()

from statsmodels.distributions.empirical_distribution import ECDF

Fhat = ECDF(nerve_firings)

plt.plot(nerve_firings, Fhat(nerve_firings), 'o')
plt.show()

alpha = 0.05
epsilon_n = np.sqrt(np.log(2/alpha)/(2*len(nerve_firings.index)))

plt.plot(nerve_firings, Fhat(nerve_firings), 'bo', label = r'Empirical CDF $\hat{F}_n(x)$' )
plt.plot(nerve_firings, np.minimum(Fhat(nerve_firings) + epsilon_n,1), 'rx', label = r'$\hat{F}_n(x)+\varepsilon_n$')
plt.plot(nerve_firings, np.maximum(Fhat(nerve_firings) - epsilon_n,0), 'r+', label = r'$\hat{F}_n(x)-\varepsilon_n$')

plt.legend(loc = 4)

plt.show()

Fhat(0.6)-Fhat(0.4)

nerve_firings.skew()

N = len(nerve_firings.index) #Number of real samples
B = 1000 #Number of bootstrap samples

theta_boot = []

for i in range(B):
    # Since nerve_firings is a Series, we can index its rows as nerve_firings[ ]
    theta_boot.append(pd.Series([nerve_firings[np.random.randint(0,N)] for j in range(N)]).skew())
theta_boot = np.array(theta_boot)

theta_boot.std()

[np.percentile(theta_boot,0.025, interpolation='higher'), np.percentile(theta_boot, 0.975)]

