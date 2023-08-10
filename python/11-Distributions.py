get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import uniform

dat = uniform.rvs(size=10000)

plt.hist(dat);

from scipy.stats import norm

dat = norm.rvs(size=10000)

plt.hist(dat, bins=20);

from scipy.stats import bernoulli

r = bernoulli.rvs(0.5, size=100000)

plt.hist(r);

from scipy.stats import gamma

dat = gamma.rvs(a=1, size=100000)

plt.hist(dat, 50);

from scipy.stats import beta

dat = beta.rvs(1, 1, size=1000)

plt.hist(dat, 50);

from scipy.stats import poisson

dat = poisson.rvs(mu=1, size=100000)

plt.hist(dat);

