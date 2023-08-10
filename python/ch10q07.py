import pandas as pd

data = pd.read_csv('../data/snodgrass.csv')

data

data.groupby('author').aggregate(['mean', 'count'])

t_obs = .022175

from itertools import permutations

obs = list(data['total'])

import numpy as np

perms = (p for p in permutations([1,2,3]))
for p in perms:
    print(np.mean(p[:2]) - np.mean(p[2:]))

# there are too many permutations to count all of them, so we set some limit N
# on the total number of observed permutations


def permutation_test(obs, m, c, N):
    i = 0 # counter for T>c
    
    # adjust m so there's at least one observation in each sample
    m = min(m, len(obs)-1)

    # evaluate permutations
    for j in range(N):
        p = np.random.permutation(obs)
        # calculate stat
        T = np.mean(p[:m]) - np.mean(p[m:])
        # count if > c
        if T > c:
            i += 1
            
    # i/N approximates the p-value as N -> N!
    return i/N

permutation_test(obs, 10, t_obs, N=100000)

X = data[data['author']=='snodgrass']['total']
Y = data[data['author']=='twain']['total']

x_bar = np.mean(X)
m = len(X)
s_x = np.std(X)

y_bar = np.mean(Y)
n = len(Y)
s_y = np.std(Y)

W = (x_bar - y_bar)/np.sqrt(s_x**2/m + s_y**2/n)

np.abs(W) > 1.96

from scipy.stats import norm

norm.cdf(W)

theta_hat = x_bar-y_bar
se_hat = np.sqrt(s_x**2/m + s_y**2/n)

(theta_hat, theta_hat+2*se_hat, theta_hat-2*se_hat)

