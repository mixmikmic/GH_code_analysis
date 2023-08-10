def func(x):
    return x[0]**2 + 2*x[1]**2 + 3*x[2]**2

def con(x):
    return x[0] + x[1] + x[2] - 3.5  # rewritten in form c <= 0

x = [1.0, 1.0, 1.0]
sigma = [0.00, 0.06, 0.2]

import numpy as np

def stats(n):
    f = np.zeros(n)
    c = np.zeros(n)

    for i in range(n):
        x1 = x[0]
        x2 = x[1] + np.random.randn(1)*sigma[1]
        x3 = x[2] + np.random.randn(1)*sigma[2]

        f[i] = func([x1, x2, x3])
        c[i] = con([x1, x2, x3])
        
    # mean
    mu = np.average(f)
    
    # standard deviation
    std = np.std(f, ddof=1)  #ddof=1 gives an unbiased estimate (np.sqrt(1.0/(n-1)*(np.sum(f**2) - n*mu**2)))

    return mu, std, f, c

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

nvec = np.logspace(1, 6, 20)
muvec = np.zeros(20)
stdvec = np.zeros(20)

for i, n in enumerate(nvec):
    muvec[i], stdvec[i], _, _ = stats(int(n))
    print i
    
plt.figure()
plt.semilogx(nvec, muvec, '-o')

plt.figure()
plt.semilogx(nvec, stdvec, '-o')
plt.show()

n = 1e5
mu, std, f, c = stats(int(n))
print 'mu =', mu
print 'sigma =', std
plt.figure()
plt.hist(f, bins=20);

reliability = np.count_nonzero(c <= 0.0)/float(n)
print 'reliability = ', reliability*100, '%'



from pyDOE import lhs
from scipy.stats.distributions import norm

def statsLHS(n):
    f = np.zeros(n)
    c = np.zeros(n)
    
    # generate latin hypercube sample points beforehand from normal dist
    lhd = lhs(2, samples=n)
    rpt = norm(loc=0, scale=1).ppf(lhd)

    for i in range(n):
        x1 = x[0]
        x2 = x[1] + rpt[i, 0]*sigma[1]
        x3 = x[2] + rpt[i, 1]*sigma[2]

        f[i] = func([x1, x2, x3])
        c[i] = con([x1, x2, x3])
        
    # mean
    mu = np.average(f)
    
    # standard deviation
    std = np.std(f, ddof=1)  #ddof=1 gives an unbiased estimate (np.sqrt(1.0/(n-1)*(np.sum(f**2) - n*mu**2)))

    return mu, std, f, c


muLHS = np.zeros(20)
stdLHS = np.zeros(20)

for i, n in enumerate(nvec):
    muLHS[i], stdLHS[i], _, _ = statsLHS(int(n))
    print i
    
plt.figure()
plt.semilogx(nvec, muvec, '-o')
plt.semilogx(nvec, muLHS, '-o')

plt.figure()
plt.semilogx(nvec, stdvec, '-o')
plt.semilogx(nvec, stdLHS, '-o')
plt.show()



