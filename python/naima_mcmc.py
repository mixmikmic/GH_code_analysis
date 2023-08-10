import naima
import numpy as np
from astropy.io import ascii
import astropy.units as u
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

hess_spectrum = ascii.read('RXJ1713_HESS_2007.dat', format='ipac')
fig = naima.plot_data(hess_spectrum)

from naima.models import ExponentialCutoffPowerLaw, InverseCompton
from naima import uniform_prior

ECPL = ExponentialCutoffPowerLaw(1e36/u.eV, 5*u.TeV, 2.7, 50*u.TeV)
IC = InverseCompton(ECPL, seed_photon_fields=['CMB', ['FIR', 30*u.K, 0.4*u.eV/u.cm**3]])

# define labels and initial vector for the parameters
labels = ['log10(norm)', 'index', 'log10(cutoff)']
p0 = np.array((34, 2.7, np.log10(30)))

# define the model function
def model(pars, data):
    ECPL.amplitude = (10**pars[0]) / u.eV
    ECPL.alpha = pars[1]
    ECPL.e_cutoff = (10**pars[2]) * u.TeV

    return IC.flux(data['energy'], distance=2.0*u.kpc), IC.compute_We(Eemin=1*u.TeV)

from naima import uniform_prior

def lnprior(pars):
    lnprior = uniform_prior(pars[1], -1, 5)
    return lnprior

sampler, pos = naima.run_sampler(data_table=hess_spectrum, model=model, prior=lnprior, p0=p0, labels=labels,
                                nwalkers=32, nburn=50, nrun=100, prefit=True, threads=4)

# inspect the chains stored in the sampler for the three free parameters
f = naima.plot_chain(sampler, 0)
f = naima.plot_chain(sampler, 1)
f = naima.plot_chain(sampler, 2)

# make a corner plot of the parameters to show covariances
f = naima.plot_corner(sampler)

# Show the fit
f = naima.plot_fit(sampler)
f.axes[0].set_ylim(bottom=1e-13)

# Inspect the metadata blob saved
f = naima.plot_blob(sampler,1, label='$W_e (E_e>1$ TeV)')

# There is also a convenience function that will plot all the above files to pngs or a single pdf
naima.save_diagnostic_plots('RXJ1713_naima_fit', sampler, blob_labels=['Spectrum','$W_e (E_e>1$ TeV)'])

suzaku_spectrum = ascii.read('RXJ1713_Suzaku-XIS.dat')
f=naima.plot_data(suzaku_spectrum)

f=naima.plot_data([suzaku_spectrum, hess_spectrum], sed=True)

#from naima.models import ExponentialCutoffPowerLaw, InverseCompton
#from naima import uniform_prior

#ECPL = ExponentialCutoffPowerLaw(1e36/u.eV, 10*u.TeV, 2.7, 50*u.TeV)
#IC = InverseCompton(ECPL, seed_photon_fields=['CMB', ['FIR', 30*u.K, 0.4*u.eV/u.cm**3]])

## define labels and initial vector for the parameters
#labels = ['log10(norm)', 'index', 'log10(cutoff)']
#p0 = np.array((34, 2.7, np.log10(30)))

## define the model function
#def model(pars, data):
#    ECPL.amplitude = (10**pars[0]) / u.eV
#    ECPL.alpha = pars[1]
#    ECPL.e_cutoff = (10**pars[2]) * u.TeV

#    return IC.flux(data['energy'], distance=2.0*u.kpc), IC.compute_We(Eemin=1*u.TeV)

#from naima import uniform_prior

#def lnprior(pars):
#    lnprior = uniform_prior(pars[1], -1, 5)
#    return lnprior

