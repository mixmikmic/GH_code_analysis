get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    print("Install seaborn. It help you make prettier figures!")

import numpy as np
from astropy.modeling import models

g = models.Gaussian1D()

# Generate fake data
np.random.seed(0)
x = np.linspace(-5., 5., 200)
y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
y += np.random.normal(0., 0.2, x.shape)
yerr = 0.2

plt.figure(figsize=(8,5))
plt.errorbar(x, y, yerr=yerr, fmt='ko')

from abc import ABCMeta
import abc

class ObjectiveFunction(object):
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def __call__(self):
        """
        Any objective function must have a `__call__` method that 
        takes parameters as a numpy-array and returns a value to be 
        optimized or sampled.
        
        """
        pass


class LogLikelihood(ObjectiveFunction):
    __metaclass__ = ABCMeta

    def __init__(self, x, y, model):
        """
        x : iterable
            x-coordinate of the data. Could be multi-dimensional.
        
        y : iterable
            y-coordinate of the data. Could be multi-dimensional.
        
        model: probably astropy.modeling.FittableModel instance
            Your model
        """
        self.x = x
        self.y = y
        
        self.model = model
        
    @abc.abstractmethod
    def evaluate(self, parameters):
        """
        This is where you define your log-likelihood. Do this!
        
        """
        pass
    
    @abc.abstractmethod
    def __call__(self, parameters):
        return self.loglikelihood(parameters)

from astropy.modeling.fitting import _fitter_to_model_params
from astropy.modeling import models

class GaussianLogLikelihood(LogLikelihood, object):
    
    def __init__(self, x, y, yerr, model):
        """
        A Gaussian likelihood.
        
        Parameters
        ----------
        x : iterable
            x-coordinate of the data
            
        y : iterable
            y-coordinte of the data
        
        yerr: iterable
            the error on the data
            
        model: an Astropy Model instance
            The model to use in the likelihood.
        
        """
        
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
        
    def evaluate(self, pars):
        _fitter_to_model_params(self.model, pars)
        
        mean_model = self.model(self.x)
        
        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) - (self.y-mean_model)**2/(2.*self.yerr**2))
        
        return loglike
    
    def __call__(self, pars):
        return self.evaluate(pars)
        
        

g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)

loglike = GaussianLogLikelihood(x, y, yerr, g_init)

loglike([1, 0, 5])

import scipy.optimize

neg_loglike = lambda x: -loglike(x)

start_params = [1, 0, 1]
opt = scipy.optimize.minimize(neg_loglike, start_params, method="L-BFGS-B", tol=1.e-10)

opt

print("The value of the negative log-likelihood: " + str(opt.fun))

opt.x

fit_pars = opt.x
_fitter_to_model_params(loglike.model, fit_pars)

plt.figure(figsize=(8,5))
plt.errorbar(x, y, yerr=yerr, fmt='ko')

plt.plot(x, loglike.model(x), lw=3)

class LogPosterior(ObjectiveFunction):
    __metaclass__ = ABCMeta
    
    def __init__(self, x, y, model):
        """
        x : iterable
            x-coordinate of the data. Could be multi-dimensional.
        
        y : iterable
            y-coordinate of the data. Could be multi-dimensional.
        
        model: probably astropy.modeling.FittableModel instance
            Your model
        """

        self.x = x
        self.y = y
        self.model = model

    @abc.abstractmethod
    def loglikelihood(self, parameters):
        pass
    
    @abc.abstractmethod        
    def logprior(self, parameters):
        pass
    
    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)
    
    def __call__(self, parameters):
        return self.logposterior(parameters)

logmin = -10000000000000000.0

class GaussianLogPosterior(LogPosterior, object):
    
    def __init__(self, x, y, yerr, model):
        
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
    def logprior(self, pars):
        """
        Some hard-coded priors.
        
        """
        # amplitude prior
        amplitude = pars[0]
        logamplitude = np.log(amplitude)
        
        logamplitude_min = -8.
        logamplitude_max = 8.0 
        p_amp = ((logamplitude_min <= logamplitude <= logamplitude_max) /                       (logamplitude_max-logamplitude_min))
        
        # mean prior
        mean = pars[1]
        
        mean_min = self.x[0]
        mean_max = self.x[-1]
        
        p_mean = ((mean_min <= mean <= mean_max) / (mean_max-mean_min))

        # width prior
        width = pars[2]
        logwidth = np.log(width)
        
        logwidth_min = -8.0
        logwidth_max = 8.0
        
        p_width = ((logwidth_min <= logwidth <= logwidth_max) / (logwidth_max-logwidth_min))

        pp = p_amp*p_mean*p_width
        
  
        if pp == 0 or np.isfinite(pp) is False:
            return logmin
        else:
            return np.log(pp)
        
    
    def loglikelihood(self, pars):
        _fitter_to_model_params(self.model, pars)
        
        mean_model = self.model(self.x)
        
        loglike = np.sum(-0.5*np.log(2.*np.pi) - np.log(self.yerr) - (self.y-mean_model)**2/(2.*self.yerr**2))

        return loglike
    

lpost = GaussianLogPosterior(x, y, yerr, g_init)
lpost([1, 0, 1])

neg_lpost = lambda x: -lpost(x)

start_params = [1, 0, 1]
opt = scipy.optimize.minimize(neg_lpost, start_params, method="L-BFGS-B", tol=1.e-10)

fit_pars = opt.x
_fitter_to_model_params(loglike.model, fit_pars)

plt.figure(figsize=(8,5))
plt.errorbar(x, y, yerr=yerr, fmt='ko')

plt.plot(x, lpost.model(x), lw=4)

from statsmodels.tools.numdiff import approx_hess
phess = approx_hess(opt.x, neg_lpost)

cov = np.linalg.inv(phess)

import emcee

# define some MCMC parameters
nwalkers = 500
ndim = opt.x.shape[0]
threads = 4
burnin = 200
niter = 200

# starting parameters for the walkers
p0 = np.array([np.random.multivariate_normal(opt.x, cov) for
               i in range(nwalkers)])

# initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lpost, threads=threads)

# run the burn-in
pos, prob, state = sampler.run_mcmc(p0, burnin)

# reset the sampler
sampler.reset()

# do the actual MCMC run
_, _, _ = sampler.run_mcmc(pos, niter, rstate0=state)

print("Acceptance fraction: " + str(np.nanmean(sampler.acceptance_fraction)))

mcall = sampler.flatchain.T
for chain in mcall:
    plt.figure(figsize=(16,6))
    plt.plot(chain)
    

import corner
corner.corner(sampler.flatchain,
              quantiles=[0.16, 0.5, 0.84],
              show_titles=False, title_args={"fontsize": 12});

