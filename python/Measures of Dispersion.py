import numpy as np
import math

np.random.seed(121)
X = np.sort(np.random.randint(100, size=20))
print 'X:', X
mu = np.mean(X)
print 'Mean of X:', mu

print 'Range of X:', np.ptp(X)

abs_dispersion = [abs(mu - x) for x in X]
MAD = sum(abs_dispersion)/len(abs_dispersion)
print 'Mean absolute deviation of X:', MAD

print 'Variance of X:', np.var(X)
print 'Standard deviation of X:', np.std(X)

k = 1.25
dist = k*np.std(X)
l = [x for x in X if abs(x - mu) <= dist]
print 'Observations within', k, 'stds of mean:', l
print 'Confirming that', float(len(l))/len(X), '>', 1 - 1/k**2

# Because there is no built-in semideviation, we'll compute it ourselves
lows = [e for e in X if e <= mu]
semivar = sum(map(lambda x: (x - mu)**2,lows))/len(lows)

print 'Semivariance of X:', semivar
print 'Semideviation of X:', math.sqrt(semivar)

B = 19
lows_B = [e for e in X if e <= B]
semivar_B = sum(map(lambda x: (x - B)**2,lows_B))/len(lows_B)

print 'Target semivariance of X:', semivar_B
print 'Target semideviation of X:', math.sqrt(semivar_B)

