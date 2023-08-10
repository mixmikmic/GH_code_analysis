from context import lens
from lens.sis.inference import *

import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import emcee

s = pd.read_clipboard()

s

s.source_id

s['phot_g_mean_mag']

19.091948/17.603382

s.phot_g_mean_mag-2.5*np.log(s.phot_g_mean_flux)

(40-18.616364)/(40-15.188815)

s.phot_g_mean_mag/s.phot_g_mean_flux_over_error

s['x']=(s.ra-s.ra.mean())*u.deg.to(u.arcsec)
s['y']=(s.dec-s.dec.mean())*u.deg.to(u.arcsec)
s['dx']=(s.pmra-s.pmra.mean())
s['dy']=(s.pmdec-s.pmdec.mean())
s['xe']=s.ra_error*u.mas.to(u.arcsec)
s['ye']=s.dec_error*u.mas.to(u.arcsec)
s['dxe']=s.pmra_error
s['dye']=s.pmdec_error
s['g'] = s.phot_g_mean_mag
s['ge'] = s.phot_g_mean_mag/s.phot_g_mean_flux_over_error

def plot_chains(sampler,warmup=400):
    fig, ax = plt.subplots(ndim,3, figsize=(12, 12))
    samples = sampler.chain[:, warmup:, :].reshape((-1, ndim))
    medians = []
    for i in range(ndim):
        ax[i,0].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2);
        ax[i,0].vlines(warmup,np.min(sampler.chain[:, :, i].T),np.max(sampler.chain[:, :, i].T),'r')
        ax[i,1].hist(samples[:,i],bins=100,label=parameter[i]);
        ax[i,1].legend()
        ax[i,1].vlines(np.median(samples[:,i]),0,10000,lw=1,color='r',label="median")
        medians.append(np.median(samples[:,i]))
        ax[i,2].hexbin(samples[:,i],samples[:,(i+1)%ndim])#,s=1,alpha=0.1);
    return medians

parameter = "xS,yS,gS,bL,xL,yL".split(',')

data = s[['x','y','g','xe','ye','ge']].as_matrix()

data

np.random.seed(0)
def init(N):
    """ to initialise each walkers initial value : sets parameter randomly """
    xS = norm.rvs(0,0.2,size=N)
    yS = norm.rvs(0,0.2,size=N)
    gS = gamma.rvs(10,5,size=N)
    xL = norm.rvs(0,0.2,size=N)
    yL = norm.rvs(0,0.2,size=N)
    bL = beta.rvs(2,3,size=N)
    return np.transpose(np.array([xS,yS,gS,bL,xL,yL]))

ndim = 6  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nsteps = 1000  # number of MCMC steps

starting_guesses = init(nwalkers)

np.std([log_prior(guess) for guess in starting_guesses])

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])
get_ipython().run_line_magic('time', 'x = sampler.run_mcmc(starting_guesses, nsteps)')

medians = plot_chains(sampler)

medians

np.random.seed(0)
def init2(N):
    """ to initialise each walkers initial value : sets parameter randomly """
    xS = norm.rvs(medians[0],0.02,size=N)
    yS = norm.rvs(medians[1],0.02,size=N)
    gS = norm.rvs(medians[2],1,size=N)
    bL = norm.rvs(medians[3],0.1,size=N)
    xL = norm.rvs(medians[4],0.02,size=N)
    yL = norm.rvs(medians[5],0.01,size=N)
    
    return np.transpose(np.array([xS,yS,gS,bL,xL,yL]))

starting_guesses2 = init2(nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data])
get_ipython().run_line_magic('time', 'x = sampler.run_mcmc(starting_guesses2, nsteps)')

plot_chains(sampler)

parameter = "xS,yS,dxS,dyS,gS,bL,xL,yL".split(',')

from lens.sis.inferencePM import *

data_pm = s[['x','y','dx','dy','g','xe','ye','dxe','dye','ge']].as_matrix()

data_pm

np.random.seed(0)
def initPM(N):
    """ to initialise each walkers initial value : sets parameter randomly """
    xS = norm.rvs(medians[0],0.02,size=N)
    yS = norm.rvs(medians[1],0.02,size=N)
    gS = norm.rvs(medians[2],1,size=N)
    bL = norm.rvs(medians[3],0.1,size=N)
    xL = norm.rvs(medians[4],0.02,size=N)
    yL = norm.rvs(medians[5],0.01,size=N)
    
    dxS = norm.rvs(0,0.1,size=N)
    dyS = norm.rvs(0,0.1,size=N)
    
    return np.transpose(np.array([xS,yS,dxS,dyS,gS,bL,xL,yL]))

ndim = 8  # number of parameters in the model
nwalkers = 50  # number of MCMC walkers
nsteps = 2000  # number of MCMC steps

starting_guesses_pm = initPM(nwalkers)

np.std([log_prior_pm(guess) for guess in starting_guesses_pm])

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_pm, args=[data_pm])
get_ipython().run_line_magic('time', 'x = sampler.run_mcmc(starting_guesses_pm, nsteps)')

medians = plot_chains(sampler,warmup=1000)

medians



