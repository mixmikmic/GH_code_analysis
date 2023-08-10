import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, beta, poisson
from numpy.random import choice
from numpy.linalg import matrix_power

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Choose our observed value of Y 
# and decide on values for the constants \theta, \mu, and \tau:
y = 3
sigma = 1
mu = 0 
tau = 2
d = 1
niter = 10**4
Theta = np.zeros(niter, dtype=np.float)   # allocate a vector Theta of length niter

# Initialize \theta to the observed value y, then run the algorithm:
Theta[0] = y
for i in range(1, niter):
    theta_new = Theta[i-1] + norm.rvs(scale=d, size=1)
    # Compute the acceptance probability:
    r = norm.pdf(y, loc=theta_new, scale=sigma) * norm.pdf(theta_new, loc=mu, scale=tau) /          (norm.pdf(y, loc=Theta[i-1], scale=sigma) * norm.pdf(Theta[i-1], loc=mu, scale=tau))
    flip = binom.rvs(n=1, p=min(r, 1), size=1)
    Theta[i] = theta_new if flip == 1 else Theta[i-1]

# Discard some of the initial draws 
# to give the chain some time to approach the stationary distribution:
Theta_latter = Theta[niter/2:]

# Create a histogram
plt.figure(figsize=(10, 5))
_ = plt.hist(Theta_latter, bins=100)

# Decide the observed value of X, as well as the constants \lambda, a, b:
x = 7
l = 10
a = 1
b = 1
niter = 10**4
P = np.zeros(niter, dtype=np.float)
N = np.zeros(niter, dtype=np.int)

# Initialize p and N to the values 0.5 and 2x, respectively,
# then run the algorithm:
P[0] = 0.5
N[0] = 2*x
for i in range(1, niter):
    P[i] = beta.rvs(a=x+a, b=N[i-1]-x+b)
    N[i] = x + poisson.rvs(mu=l*(1-P[i-1]))

# Discard some of the initial draws 
# to give the chain some time to approach the stationary distribution:
P_latter = P[niter/2:]
N_latter = N[niter/2:]

# Create a histogram
plt.figure(figsize=(10, 10))
fig, axes = plt.subplots(2, 1)
_ = axes[0].hist(P_latter, bins=100)
_ = axes[1].hist(N_latter, bins=100)



