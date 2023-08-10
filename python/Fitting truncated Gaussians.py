from scipy import stats
import scipy.sparse
import numpy as np
import maxentropy
import maxentropy.skmaxent
import maxentropy.maxentutils

whichplot = 2  # sub-plot in Figure 6.1 (0 , 1 , or 2)
d = 1     # number of dimensions
m = d*3   # number of features

# Bounds
o = np.ones(d)
if whichplot == 0:
    lower = -2.5 * o
    upper = -lower
elif whichplot == 1:
    lower = 0.5 * o
    upper = 2.5 * o
elif whichplot == 2:
    lower = -0.1 * o
    upper = 0.1 * o

def f0(x):
    return x

def f1(x):
    return x**2

def f2(x):
    return (lower < x) & (x < upper)

f = [f0, f1, f2]

# Target constraint values
b = np.empty (m , float )
if whichplot == 0:
    b [0: m :3] = 0   # expectation
    b [1: m :3] = 1   # second moment
    b [2: m :3] = 1   # truncate completely outside bounds
elif whichplot == 1:
    b [0: m :3] = 1.0 # expectation
    b [1: m :3] = 1.2 # second moment
    b [2: m :3] = 1   # truncate completely outside bounds
elif whichplot == 2:
    b [:] = [0. , 0.0033 , 1]

b

from scipy.stats import norm

mu = b[0]
sigma = (b[1] - mu**2)**0.5
mu, sigma

auxiliary = stats.norm(loc=mu, scale=sigma)

auxiliary

from maxentropy.maxentutils import auxiliary_sampler_scipy

sampler = auxiliary_sampler_scipy(auxiliary, n=10**5)

xs, log_q_xs = sampler()

xs.shape, log_q_xs.shape

xs

xs[:10, 0]

f

b

model = maxentropy.skmaxent.MCMinDivergenceModel(f, sampler)   # create a model

k = np.reshape(b, (1, -1))
k.shape

model.fit(k)

model.expectations()

np.allclose(model.expectations(), k)

model.iters, model.fnevals

get_ipython().magic('matplotlib inline')

lower, upper

# Plot the marginal pdf in dimension 1, letting x_d =0
# for all other dimensions d.
xs = np.arange(lower[0], upper[0], (upper[0] - lower[0]) / 100.)
all_xs = np.zeros((len(xs), d), float)
all_xs[:, 0] = xs

all_xs.shape

model.features(xs)

pdf = model.pdf(model.features(all_xs))

xs.shape

import matplotlib.pyplot as plt
plt.plot(xs, pdf)
plt.ylim(0, pdf.max()*1.1)

model.expectations()

b

np.allclose(model.expectations(), b, atol=1e-6)

