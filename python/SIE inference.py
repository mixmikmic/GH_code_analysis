get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import astropy.units as u

mp.rcParams['figure.figsize'] = (12, 8)

from context import lens
from lens.sie.inference import *

import emcee

plt.subplot(221)
x = np.arange(-1,30,0.01)
y = [magnitudePrior(v) for v in x]
plt.plot(x,y)
plt.xlabel("Magnitude radius prior")

plt.subplot(221)
x = np.arange(-1,20,0.01)
y = [radiusPrior(v) for v in x]
plt.plot(x,y)
plt.xlabel("Einstein's radius prior")

plt.subplot(222)
x = np.arange(-1,2,0.01)
y = [ratioPrior(v) for v in x]
plt.plot(x,y)
plt.xlabel("Lens axis ratio prior")

plt.subplot(223)
x = np.arange(-10,10,0.01)
y = [positionPrior(v) for v in x]
plt.plot(x,y)
plt.xlabel("Position prior")

plt.subplot(224)
x = np.arange(-1,7,0.01)
y = [thetaPrior(v) for v in x]
plt.plot(x,y)
plt.xlabel("Lens orientation prior")

model = np.array([0.1,0.1,18,2,0.5,0,0,0.])
parameter = "xS,yS,gS,bL,qL,xL,yL,thetaL".split(',')

log_prior(model)

error = np.concatenate((np.ones((4,2))*0.001,np.ones((4,1))*0.01),axis=1)

data = np.concatenate((np.array(getImages(model)),error),axis=1)

np.around(data,3)

log_likelihood(model,data)

log_posterior(model,data)

ndim = len(model)  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nsteps = 1000  # number of MCMC steps

np.random.seed(0)
def init(N):
    """ to initialise each walkers initial value : sets parameter randomly """
    xs = norm.rvs(0,0.2,size=N)
    ys = norm.rvs(0,0.2,size=N)
    gs = gamma.rvs(10,5,size=N)
    xl = norm.rvs(0,0.2,size=N)
    yl = norm.rvs(0,0.2,size=N)
    b = 2*beta.rvs(2,3,size=N)
    q =  np.random.uniform(0,1,N)
    theta = np.random.uniform(0,np.pi,N)
    return np.transpose(np.array([xs,ys,gs,b,q,xl,yl,theta]))

starting_guesses = init(nwalkers)

np.std([log_prior(guess) for guess in starting_guesses])

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])
get_ipython().run_line_magic('time', 'x = sampler.run_mcmc(starting_guesses, nsteps)')

def plot_chains(sampler,warmup=100):
    fig, ax = plt.subplots(ndim,3, figsize=(12, 12))
    samples = sampler.chain[:, warmup:, :].reshape((-1, ndim))
    for i in range(ndim):
        ax[i,0].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
        ax[i,0].vlines(warmup,np.min(sampler.chain[:, :, i].T),np.max(sampler.chain[:, :, i].T),'r')
        ax[i,1].hist(samples[:,i],bins=100,label=parameter[i]);
        ax[i,1].legend()
        ax[i,1].vlines(np.median(samples[:,i]),0,10000,lw=1,color='r',label="median")
        ax[i,1].vlines(np.median(model[i]),0,5000,lw=1,color='b',label="true")
        ax[i,2].hexbin(samples[:,i],samples[:,(i+1)%ndim])#,s=1,alpha=0.1);
plot_chains(sampler)

np.random.seed(0)
def init2(N):
    """ to initialise each walkers initial value : sets parameter randomly """
    xs = norm.rvs(0.1,0.05,size=N)
    ys = norm.rvs(0.1,0.05,size=N)
    gs = norm.rvs(18,0.5,size=N)
    xl = norm.rvs(0,0.05,size=N)
    yl = norm.rvs(0,0.05,size=N)
    b  = norm.rvs(2,0.1,size=N)
    q  = norm.rvs(0.5,0.1,size=N)
    theta = np.random.uniform(0,np.pi,N)
    return np.transpose(np.array([xs,ys,gs,b,q,xl,yl,theta]))

starting_guesses = init2(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])
get_ipython().run_line_magic('time', 'x = sampler.run_mcmc(starting_guesses, nsteps)')

plot_chains(sampler)



