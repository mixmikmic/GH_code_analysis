# General imports
get_ipython().magic('matplotlib inline')

import logging
import numpy as np
import pylab as plt
from scipy import stats
from scipy import integrate
from scipy.integrate import simps,trapz,quad,nquad
from scipy.interpolate import interp1d
from scipy.misc import factorial

# Constants
MMIN,MMAX  = 4e6,4e9
MLOW,MHIGH = 0.3e8,4e9
P = (MMIN,MMAX,MLOW,MHIGH)
NSTEPS=(1000,1000)

# Defaults                                                                          
ALPHA=1.9
FRAC=0.02
MHALO=1e11
SIGMA = 0

# Utility functions

def create_mass_array(log=True,nsteps=(1500,1300)):
    """ Create an array spanning the true and observable mass ranges.                
                                                                                     
    Parameters:                                                                      
    -----------                                                                      
    p      : Tuple of the range of masses (MMIN,MMAX,MHIGH,MLOW)                     
    nsteps : Number of steps to span the ranges (NTRUE, NCONV)                       
    log    : Sample in log or linear space                                           
                                                                                     
    Returns:                                                                         
    --------                                                                         
    m,mp,mm,mmp : The                                                                
    """
    nsteps = map(int,nsteps)
    if log:
        m = np.logspace(np.log10(MMIN),np.log10(MMAX),nsteps[0])
        mp = np.logspace(np.log10(MLOW),np.log10(MHIGH),nsteps[1])
    else:
        m = np.linspace(MMIN,MMAX,nsteps[0])
        mp = np.linspace(MLOW,MHIGH,nsteps[1])
    mm,mmp = np.meshgrid(m,mp)
    return m,mp,mm,mmp

def mhalo(radius=None):
    """ Return the halo mass as a function of maximum radius.                        
    WARNING: Returns constant MHALO independent of R!                                
                                                                                     
    Parameters:                                                                      
    -----------                                                                      
    radius : Maximum radius for inclused halo mass                                   
                                                                                     
    Returns:                                                                         
    --------                                                                         
    mhalo : Enclosed halo mass                                                       
    """
    return MHALO

def dP_dm_true(m,alpha):
    """ True mass function (Eqn. 6) normalized over full mass range [MMIN,MMAX].
    
    Parameters:
    ---------- 
    m     : True mass of subhalo
    alpha : Power-law index of subhalo mass function

    Returns:
    --------                                               
    dP_dm_true : Normalized pdf                                                      
    """
    m = np.atleast_1d(m)
    ret = ((1-alpha)*m**(-alpha))/(MMAX**(1-alpha)-MMIN**(1-alpha))
    ret = np.where(alpha==1,(m**-alpha)/np.log(MMAX/MMIN),ret)
    return np.where(np.isfinite(ret),ret,np.nan)

def dP_dm_conv(m,mp,alpha,sigma=SIGMA):
    """ The convolved mass function.
                                                                                     
    Parameters:                                                                      
    -----------                                                                      
    m   : The range of true masses                                                   
    mp  : The range of observed masses                                               
                                                                                     
    Returns:                                                                         
    --------                                                                         
    dP_dm_conv : The integrated convolved mass function                              
    """
    if sigma == 0:
        # Convolution replaced with delta function when sigma == 0
        return dP_dm_true(np.atleast_2d(mp.T)[0],alpha)
    else:
        return simps(dP_dm_true(m,alpha)*stats.norm.pdf(m,loc=mp,scale=sigma),m)

def mu0(alpha, frac, radius=None):
    """ Expected number of substructures from the true mass function (Eq. 5).        
                                                                                     
    Parameters:                                                                      
    -----------                                                                      
    alpha : Slope of the substructure mass function                                  
    frac  : Substructure mass fraction                                               
    radius: Enclosed radius                                                          
                                                                                     
    Returns:                                                                         
    --------                                                                         
    mu0   : Predicted number of substructures for the true mass function             
    """
    alpha = np.atleast_1d(alpha)
    integral = ( (2-alpha)*(MMAX**(1-alpha) - MMIN**(1-alpha))) /                ( (1-alpha)*(MMAX**(2-alpha) - MMIN**(2-alpha)))
    integral = np.where(alpha==2,-(MMAX**-1 - MMIN**-1)/np.log(MMAX/MMIN),integral)
    integral = np.where(alpha==1,np.log(MMAX/MMIN)/(MMAX - MMIN),integral)
    return frac * mhalo(radius) * integral

def mu(alpha, frac, radius=None, sigma=SIGMA):
    """ Expected number of substructures from the observable mass function (Eq. 4)   
                                                                                     
    Parameters:                                                                      
    -----------                                                                      
    alpha : Slope of the substructure mass function                                  
    frac  : Substructure mass fraction                                               
    radius: Enclosed radius                                                          
    sigma : Substructure mass error                                                  
                                                                                     
    Returns:                                                                         
    --------                                                                         
    mu   : Predicted number of substructures for the observable mass function        
    """
    m,mp,mm,mmp = create_mass_array()
    _mu0 = mu0(alpha, frac, radius)
    _integral = simps(dP_dm_conv(mm,mmp,alpha,sigma=sigma),mp)
    return _mu0 * _integral

def LogProbNumber(data, alpha, frac, R=1, sigma=SIGMA):
    """ Logarithm of the joint probability for the number of substructures.          
                                                                                     
    Parameters:                                                                      
    -----------                                                                      
    data : Input data                                                                
    alpha: Index of the mass function                                                
    frac : Substructure mass fraction                                                
                                                                                     
    Returns:                                                                         
    --------                                                                         
    prob : Logarithm of the joint Poission probability                               
    """
    logging.debug(' LogProbNumber: %s'%len(data))
    nsrc = data['nsrc']
    _mu = mu(alpha,frac,R,sigma=sigma)
    return np.sum(stats.poisson.logpmf(nsrc[:,np.newaxis],_mu),axis=0)

def LogProbMass(data, alpha, sigma=SIGMA):
    """ Logarithm of the joint probability for mass of substructures.                
                                                                                     
    Parameters:                                                                      
    -----------                                                                      
    data : Input data                                                                
    alpha: Index of the mass function                                                
                                                                                     
    Returns:                                                                         
    --------                                                                         
    prob: Logarithm of the joint spectral probability                                
    """
    logging.debug(' LogProbMass: %s'%len(data))
    m,mp,mm,mmp = create_mass_array()
    masses = np.concatenate(data['mass'])
    top = np.sum(np.log([dP_dm_conv(m,mi,alpha,sigma=sigma) for mi in masses]))
    bottom = len(masses)*np.log(simps(dP_dm_conv(mm,mmp,alpha,sigma=sigma),mp))
    return top - bottom

def LogLike(data, alpha, frac, sigma=SIGMA):
    """ Logarithm of the joint likelihood over all lens systems.                     
                                                                                     
    """
    logging.debug('LogLike: %s'%len(data))
    logpois = LogProbNumber(data, alpha, frac, sigma=sigma)
    logprob = LogProbMass(data, alpha, sigma=sigma)
    return logpois + logprob

def sample(size,alpha=ALPHA):
    """ Random samples of the mass function.

    Parameters:
    -----------
    size  : Number of smaples to make
    alpha : Index of the mass function

    Returns:
    --------
    mass : Random samples of the mass function
    """
    x = create_mass_array(log=False,nsteps=(1e4,1e1))[0]
    pdf = dP_dm_true(x,alpha)
    size = int(size)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0.)
    cdf /= cdf[-1]

    icdf = interp1d(cdf, range(0, len(cdf)), bounds_error=False, fill_value=-1)
    u = np.random.uniform(size=size)
    index = np.floor(icdf(u)).astype(int)
    index = index[index >= 0]
    masses = x[index]
    return masses

def simulate(nlens=1, alpha=ALPHA, frac=FRAC, sigma=SIGMA):
    """Generate the simulated data set of lens, sources, and masses.

    Parameters:
    -----------
    nlens: Number of lenses to generate.
    alpha: Index of the substructure mass function
    frac:  Substructure mass fraction

    Returns:
    --------
    data : Array of output lenses and substructures
    """
    # First, figure out how many lenses we are sampling
    m,mp,mm,mmp = create_mass_array()
    pdf = dP_dm_true(m,alpha)
    _mu = mu0(alpha,frac)
    lenses = stats.poisson.rvs(_mu,size=nlens)
    out = []
    for i,l in enumerate(lenses):
        masses = sample(l,alpha=alpha)
        if sigma != 0:
            masses += stats.norm.rvs(size=len(masses),scale=sigma)
        sel = (masses > MLOW) & (masses < MHIGH)
        mass = masses[sel]
        out += [(i,len(mass),mass)]

    names = ['lens','nsrc','mass']
    return np.rec.fromrecords(out,names=names)

# Simulate a large set of lenses
data = simulate(1000, alpha=ALPHA, frac=FRAC, sigma=SIGMA)

# Plot a histogram of the masses
bins = np.logspace(np.log10(MLOW),np.log10(MHIGH),50)
masses = np.concatenate(data['mass'])
n,b,p = plt.hist(masses,bins=bins,log=True,normed=True, label='Samples'); plt.gca().set_xscale('log')

# Plot the pdf normalized over the observable mass range
m,mp,mm,mmp = create_mass_array()
norm = simps(dP_dm_conv(mm,mmp,ALPHA,sigma=SIGMA),mp)
plt.plot(b,dP_dm_true(b,alpha=ALPHA)/norm,label='Normalized PDF')

plt.legend(loc='upper right')
plt.xlabel(r"Mass ($M_\odot$)"); plt.ylabel("Normalized Counts")

FRAC=0.005; ALPHA=1.9; MLOW=0.3e8; SIGMA=0
nlens=10; seed = 1
np.random.seed(seed)
fracs = np.linspace(0.001,0.03,151)
alphas = np.linspace(1.0,3.0,51)

data = simulate(nlens,alpha=ALPHA, frac=FRAC, sigma=SIGMA)
loglikes = np.array([LogLike(data,a,fracs) for a in alphas])
loglikes -= loglikes.max()
loglikes = loglikes.T

# Note the typo in VK09's definition of the 3 sigma p-value
levels = -stats.chi2.isf([0.0028,0.05,0.32,1.0],2)/2.
plt.contourf(alphas,fracs,loglikes,levels=levels,cmap='binary')
plt.axvline(ALPHA,ls='--',c='dodgerblue')
plt.axhline(FRAC,ls='--',c='dodgerblue')
plt.colorbar(label=r'$\Delta \log {\cal L}$')
plt.xlabel(r'Slope ($\alpha$)')
plt.ylabel(r'Mass Fraction ($f$)')

FRAC=0.005; ALPHA=1.9; SIGMA=0
nlens=200; seed = 0

fracs = np.linspace(0.001,0.03,151)
alphas = np.linspace(1.0,3.0,51)
fig,axes = plt.subplots(1,3,figsize=(10,3),sharey=True)
for i,m in enumerate([0.3e8,1.0e8,3e8]):
    MLOW = m
    data = simulate(nlens,alpha=ALPHA, frac=FRAC, sigma=SIGMA)
    loglikes = np.array([LogLike(data,a,fracs) for a in alphas])
    loglikes -= loglikes.max()
    loglikes = loglikes.T
    plt.sca(axes[i])
    plt.contourf(alphas,fracs,loglikes,levels=levels,cmap='binary')
    plt.axvline(ALPHA,ls='--',c='dodgerblue')
    plt.axhline(FRAC,ls='--',c='dodgerblue')
    plt.xlabel(r'Slope ($\alpha$)')
    if i == 0: plt.ylabel(r'Mass Fraction ($f$)')

