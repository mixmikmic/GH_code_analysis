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
import pymc3 as pm
import scipy.stats as stats

import logging
_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.ERROR)

n = 100
h = 61
alpha = 2
beta = 2

niter = 1000
with pm.Model() as model: # context management
    # define priors
    p = pm.Beta('p', alpha=alpha, beta=beta)
    
    # define likelihood
    y = pm.Binomial('y', n=n, p=p, observed=h)
    
    # inference
    start = pm.find_MAP() # Use MAP estimate (optimization) as the initial state for MCMC
    step = pm.Metropolis() # Have a choice of samplers
    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)

plt.hist(trace['p'], 15, histtype='step', normed=True, label='post');
x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, alpha, beta), label='prior');
plt.legend(loc='best');

# generate observed data
N = 100
_mu = np.array([10])
_sigma = np.array([2])
y = np.random.normal(_mu, _sigma, N)

niter = 1000
with pm.Model() as model:
    # define priors
    mu = pm.Uniform('mu', lower=0, upper=100, shape=_mu.shape)
    sigma = pm.Uniform('sigma', lower=0, upper=10, shape=_sigma.shape)
    
    # define likelihood
    y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)
    
    # inference
    start = pm.find_MAP()
    step = pm.Slice()
    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); 
plt.hist(trace['mu'][-niter/2:,0], 25, histtype='step');
plt.subplot(1,2,2); 
plt.hist(trace['sigma'][-niter/2:,0], 25, histtype='step');

# observed data
n = 11
_a = 6
_b = 2
x = np.linspace(0, 1, n)
y = _a*x + _b + np.random.randn(n)

with pm.Model() as model:
    a = pm.Normal('a', mu=0, sd=20)
    b = pm.Normal('b', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=20)
    
    y_est = a*x + b # simple auxiliary variables
    
    likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
    # inference
    start = pm.find_MAP()
    step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
    pm.traceplot(trace);

data = dict(x=x, y=y)

with pm.Model() as model:
    pm.glm.glm('y ~ x', data)
    step = pm.NUTS() 
    trace = pm.sample(2000, step, progressbar=True) 

pm.traceplot(trace);

plt.figure(figsize=(7, 7))
plt.scatter(x, y, s=30, label='data')
pm.glm.plot_posterior_predictive(trace, samples=100, 
                                 label='posterior predictive regression lines', 
                                 c='blue', alpha=0.2)
plt.plot(x, _a*x + _b, label='true regression line', lw=3., c='red')
plt.legend(loc='best');

# observed data
df = pd.read_csv('HtWt.csv')
df.head()

niter = 1000
with pm.Model() as model:
    pm.glm.glm('male ~ height + weight', df, family=pm.glm.families.Binomial()) 
    trace = pm.sample(niter, step=pm.Slice(), random_seed=123, progressbar=True)

# note that height and weigth in trace refer to the coefficients

df_trace = pm.trace_to_dataframe(trace)
pd.scatter_matrix(df_trace[-1000:], diagonal='kde');

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(df_trace.ix[-1000:, 'height'], linewidth=0.7)
plt.subplot(122)
plt.plot(df_trace.ix[-1000:, 'weight'], linewidth=0.7);

niter = 1000
with pm.Model() as model:
    pm.glm.glm('male ~ height + weight', df, family=pm.glm.families.Binomial()) 
    trace = pm.sample(niter, step=pm.NUTS(), random_seed=123, progressbar=True)

df_trace = pm.trace_to_dataframe(trace)
pd.scatter_matrix(df_trace[-1000:], diagonal='kde');

pm.summary(trace);

import seaborn as sn
sn.kdeplot(trace['weight'], trace['height'])
plt.xlabel('Weight', fontsize=20)
plt.ylabel('Height', fontsize=20)
plt.style.use('ggplot')

intercept, height, weight = df_trace[-niter//2:].mean(0)

def predict(w, h, height=height, weight=weight):
    """Predict gender given weight (w) and height (h) values."""
    v = intercept + height*h + weight*w
    return np.exp(v)/(1+np.exp(v))

# calculate predictions on grid
xs = np.linspace(df.weight.min(), df.weight.max(), 100)
ys = np.linspace(df.height.min(), df.height.max(), 100)
X, Y = np.meshgrid(xs, ys)
Z = predict(X, Y)

plt.figure(figsize=(6,6))

# plot 0.5 contour line - classify as male if above this line
plt.contour(X, Y, Z, levels=[0.5])

# classify all subjects
colors = ['lime' if i else 'yellow' for i in df.male]
ps = predict(df.weight, df.height)
errs = ((ps < 0.5) & df.male) |((ps >= 0.5) & (1-df.male))
plt.scatter(df.weight[errs], df.height[errs], facecolors='red', s=150)
plt.scatter(df.weight, df.height, facecolors=colors, edgecolors='k', s=50, alpha=1);
plt.xlabel('Weight', fontsize=16)
plt.ylabel('Height', fontsize=16)
plt.title('Gender classification by weight and height', fontsize=16)
plt.tight_layout();

# observed data
n = 5 * np.ones(4)
x = np.array([-0.896, -0.296, -0.053, 0.727])
y = np.array([0, 1, 3, 5])

def invlogit(x):
    return pm.exp(x) / (1 + pm.exp(x))

with pm.Model() as model:
    # define priors
    alpha = pm.Normal('alpha', mu=0, sd=5)
    beta = pm.Flat('beta')
    
    # define likelihood
    p = invlogit(alpha + beta*x)
    y_obs = pm.Binomial('y_obs', n=n, p=p, observed=y)
    
    # inference
    start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)

np.exp

f = lambda a, b, xp: np.exp(a + b*xp)/(1 + np.exp(a + b*xp))

xp = np.linspace(-1, 1, 100)
a = trace.get_values('alpha').mean()
b = trace.get_values('beta').mean()
plt.plot(xp, f(a, b, xp))
plt.scatter(x, y/5, s=50);
plt.xlabel('Log does of drug')
plt.ylabel('Risk of death');

radon = pd.read_csv('radon.csv')[['county', 'floor', 'log_radon']]
radon.head()

county = pd.Categorical(radon['county']).codes

with pm.Model() as hm:
    # County hyperpriors
    mu_a = pm.Normal('mu_a', mu=0, tau=1.0/100**2)
    sigma_a = pm.Uniform('sigma_a', lower=0, upper=100)
    mu_b = pm.Normal('mu_b', mu=0, tau=1.0/100**2)
    sigma_b = pm.Uniform('sigma_b', lower=0, upper=100)
    
    # County slopes and intercepts
    a = pm.Normal('slope', mu=mu_a, sd=sigma_a, shape=len(set(county)))
    b = pm.Normal('intercept', mu=mu_b, tau=1.0/sigma_b**2, shape=len(set(county)))
    
    # Houseehold errors
    sigma = pm.Gamma("sigma", alpha=10, beta=1)
    
    # Model prediction of radon level
    mu = a[county] + b[county] * radon.floor.values
    
    # Data likelihood
    y = pm.Normal('y', mu=mu, sd=sigma, observed=radon.log_radon)

with hm:
    start = pm.find_MAP()
    step = pm.NUTS(scaling=start)
    hm_trace = pm.sample(2000, step, start=start, random_seed=123, progressbar=True)

plt.figure(figsize=(8, 60))
pm.forestplot(hm_trace, vars=['slope', 'intercept']);



