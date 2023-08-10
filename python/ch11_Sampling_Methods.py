import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

from prml.distributions import GaussianDistribution, UniformDistribution
from prml.sampling_methods import metropolis, metropolis_hastings, rejection_sampling, sir

def func(x):
    return np.exp(-x ** 2) + 3 * np.exp(-(x - 3) ** 2)
x = np.linspace(-5, 10, 100)
dist = GaussianDistribution(mean=np.array([2.]), var=np.array([[2.]]))
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.plot(x, 15 * dist.proba(x), label=r"$kq(z)$")
plt.fill_between(x, func(x), 15 * dist.proba(x), color="gray")
plt.legend(fontsize=15)
plt.show()

samples = rejection_sampling(func, dist, k=15, n=100)
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.hist(samples, normed=True, alpha=0.2)
plt.scatter(samples, np.random.normal(scale=.03, size=(100, 1)), s=5, label="samples")
plt.legend(fontsize=15)
plt.show()

samples = sir(func, dist, n=100)
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.hist(samples, normed=True, alpha=0.2)
plt.scatter(samples, np.random.normal(scale=.03, size=(100, 1)), s=5, label="samples")
plt.legend(fontsize=15)
plt.show()

samples = metropolis(func, GaussianDistribution(mean=np.zeros(1), var=np.ones((1, 1))), n=100, downsample=10)
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.hist(samples, normed=True, alpha=0.2)
plt.scatter(samples, np.random.normal(scale=.03, size=(100, 1)), s=5, label="samples")
plt.legend(fontsize=15)
plt.show()

samples = metropolis_hastings(func, GaussianDistribution(mean=np.ones(1), var=np.ones((1, 1))), n=100, downsample=10)
plt.plot(x, func(x), label=r"$\tilde{p}(z)$")
plt.hist(samples, normed=True, alpha=0.2)
plt.scatter(samples, np.random.normal(scale=.03, size=(100, 1)), s=5, label="samples")
plt.legend(fontsize=15)
plt.show()



