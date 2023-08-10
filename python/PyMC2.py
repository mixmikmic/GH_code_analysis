from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
get_ipython().magic('precision 4')
plt.style.use('ggplot')

np.random.seed(1234)
import pymc
import scipy.stats as stats

n = 100
h = 61
alpha = 2
beta = 2

p = pymc.Beta('p', alpha=alpha, beta=beta)
y = pymc.Binomial('y', n=n, p=p, value=h, observed=True)
m = pymc.Model([p, y])

mc = pymc.MCMC(m, )
mc.sample(iter=11000, burn=10000)
plt.hist(p.trace(), 15, histtype='step', normed=True, label='post');
x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, alpha, beta), label='prior');
plt.legend(loc='best');

p = pymc.TruncatedNormal('p', mu=0.3, tau=10, a=0, b=1)
y = pymc.Binomial('y', n=n, p=p, value=h, observed=True)
m = pymc.Model([p, y])

mc = pymc.MCMC(m)
mc.sample(iter=11000, burn=10000)
plt.hist(p.trace(), 15, histtype='step', normed=True, label='post');
a, b = plt.xlim()
x = np.linspace(0, 1, 100)
a, b = (0 - 0.3) / 0.1, (1 - 0.3) / 0.1
plt.plot(x, stats.truncnorm.pdf(x, a, b, 0.3, 0.1), label='prior');
plt.legend(loc='best');

# generate observed data
N = 100
y = np.random.normal(10, 2, N)

# define priors
mu = pymc.Uniform('mu', lower=0, upper=100)
tau = pymc.Uniform('tau', lower=0, upper=1)
    
# define likelihood
y_obs = pymc.Normal('Y_obs', mu=mu, tau=tau, value=y, observed=True)
    
# inference
m = pymc.Model([mu, tau, y])
mc = pymc.MCMC(m)
mc.sample(iter=11000, burn=10000)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.hist(mu.trace(), 15, histtype='step', normed=True, label='post');
plt.legend(loc='best');
plt.subplot(122)
plt.hist(np.sqrt(1.0/tau.trace()), 15, histtype='step', normed=True, label='post');
plt.legend(loc='best');

# observed data
n = 21
a = 6
b = 2
sigma = 2
x = np.linspace(0, 1, n)
y_obs = a*x + b + np.random.normal(0, sigma, n)
data = pd.DataFrame(np.array([x, y_obs]).T, columns=['x', 'y'])

data.plot(x='x', y='y', kind='scatter', s=50);

# define priors
a = pymc.Normal('slope', mu=0, tau=1.0/10**2)
b = pymc.Normal('intercept', mu=0, tau=1.0/10**2)
tau = pymc.Gamma("tau", alpha=0.1, beta=0.1)

# define likelihood
@pymc.deterministic
def mu(a=a, b=b, x=x):
    return a*x + b

y = pymc.Normal('y', mu=mu, tau=tau, value=y_obs, observed=True)

# inference
m = pymc.Model([a, b, tau, x, y])
mc = pymc.MCMC(m)
mc.sample(iter=11000, burn=10000)

abar = a.stats()['mean']
bbar = b.stats()['mean']
data.plot(x='x', y='y', kind='scatter', s=50);
xp = np.array([x.min(), x.max()])
plt.plot(a.trace()*xp[:, None] + b.trace(), c='red', alpha=0.01)
plt.plot(xp, abar*xp + bbar, linewidth=2, c='red');

pymc.Matplot.plot(mc)

# define invlogit function
def invlogit(x):
    return pymc.exp(x) / (1 + pymc.exp(x))

# observed data
n = 5 * np.ones(4)
x = np.array([-0.896, -0.296, -0.053, 0.727])
y_obs = np.array([0, 1, 3, 5])

# define priors
alpha = pymc.Normal('alpha', mu=0, tau=1.0/5**2)
beta = pymc.Normal('beta', mu=0, tau=1.0/10**2)

# define likelihood
p = pymc.InvLogit('p', alpha + beta*x)
y = pymc.Binomial('y_obs', n=n, p=p, value=y_obs, observed=True)

# inference
m = pymc.Model([alpha, beta, y])
mc = pymc.MCMC(m)
mc.sample(iter=11000, burn=10000)

beta.stats()

xp = np.linspace(-1, 1, 100)
a = alpha.stats()['mean']
b = beta.stats()['mean']
plt.plot(xp, invlogit(a + b*xp).value)
plt.scatter(x, y_obs/5, s=50);
plt.xlabel('Log does of drug')
plt.ylabel('Risk of death');

pymc.Matplot.plot(mc)

radon = pd.read_csv('radon.csv')[['county', 'floor', 'log_radon']]
radon.head()

def make_model(x, y):
    # define priors
    a = pymc.Normal('slope', mu=0, tau=1.0/10**2)
    b = pymc.Normal('intercept', mu=0, tau=1.0/10**2)
    tau = pymc.Gamma("tau", alpha=0.1, beta=0.1)

    # define likelihood
    @pymc.deterministic
    def mu(a=a, b=b, x=x):
        return a*x + b

    y = pymc.Normal('y', mu=mu, tau=tau, value=y, observed=True)

    return locals()

plt.scatter(radon.floor, radon.log_radon)
plt.xticks([0, 1], ['Basement', 'No basement'], fontsize=20);

m = pymc.Model(make_model(radon.floor, radon.log_radon))
mc = pymc.MCMC(m)
mc.sample(iter=1100, burn=1000)

abar = mc.stats()['slope']['mean']
bbar = mc.stats()['intercept']['mean']
radon.plot(x='floor', y='log_radon', kind='scatter', s=50);
xp = np.array([0, 1])
plt.plot(mc.trace('slope')()*xp[:, None] + mc.trace('intercept')(), c='red', alpha=0.1)
plt.plot(xp, abar*xp + bbar, linewidth=2, c='red');

n = 0
i_as = []
i_bs = []
for i, group in radon.groupby('county'):

    m = pymc.Model(make_model(group.floor, group.log_radon))
    mc = pymc.MCMC(m)
    mc.sample(iter=1100, burn=1000)

    abar = mc.stats()['slope']['mean']
    bbar = mc.stats()['intercept']['mean']
    group.plot(x='floor', y='log_radon', kind='scatter', s=50);
    xp = np.array([0, 1])
    plt.plot(mc.trace('slope')()*xp[:, None] + mc.trace('intercept')(), c='red', alpha=0.1)
    plt.plot(xp, abar*xp + bbar, linewidth=2, c='red');
    plt.title(i)
    
    n += 1
    if n > 3:
        break

county = pd.Categorical(radon['county']).codes

# County hyperpriors
mu_a = pymc.Normal('mu_a', mu=0, tau=1.0/100**2)
sigma_a = pymc.Uniform('sigma_a', lower=0, upper=100)
mu_b = pymc.Normal('mu_b', mu=0, tau=1.0/100**2)
sigma_b = pymc.Uniform('sigma_b', lower=0, upper=100)

# County slopes and intercepts
a = pymc.Normal('slope', mu=mu_a, tau=1.0/sigma_a**2, size=len(set(county)))
b = pymc.Normal('intercept', mu=mu_b, tau=1.0/sigma_b**2, size=len(set(county)))

# Houseehold priors
tau = pymc.Gamma("tau", alpha=0.1, beta=0.1)

@pymc.deterministic
def mu(a=a, b=b, x=radon.floor):
    return a[county]*x + b[county]

y = pymc.Normal('y', mu=mu, tau=tau, value=radon.log_radon, observed=True)

m = pymc.Model([y, mu, tau, a, b])
mc = pymc.MCMC(m)
mc.sample(iter=110000, burn=100000)

abar = a.stats()['mean']
bbar = b.stats()['mean']

xp = np.array([0, 1])
for i, (a, b) in enumerate(zip(abar, bbar)):
    plt.figure()
    idx = county == i
    plt.scatter(radon.floor[idx], radon.log_radon[idx])
    plt.plot(xp, a*xp + b, c='red');
    plt.title(radon.county[idx].unique()[0])
    if i >= 3:
        break



