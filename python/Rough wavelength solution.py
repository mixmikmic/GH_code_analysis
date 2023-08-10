from astropy.modeling.functional_models import Voigt1D
from astropy.constants import c
from astropy.io import fits
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit, minimize

pix, wvln = np.loadtxt("../data/mdm-spring-2017/quickreduce/rough_wavelength.txt").T

plt.plot(pix, wvln, marker='o')

wvln2pix = InterpolatedUnivariateSpline(wvln[wvln.argsort()], pix[wvln.argsort()], k=3)
pix2wvln = InterpolatedUnivariateSpline(pix, wvln, k=3)

halpha_idx = int(wvln2pix(6563.))

(0.022/6563 * c).to(u.km/u.s)

(30.*u.km/u.s / c * 6563.*u.angstrom).to(u.angstrom)

wvln2pix(6563.)

pix2wvln(684.7+1)-pix2wvln(684.7)

hdu1 = fits.open("/Users/adrian/projects/gaia-wide-binaries/data/mdm-spring-2017/n3/n3.0104.fit")[0]
hdu2 = fits.open("/Users/adrian/projects/gaia-wide-binaries/data/mdm-spring-2017/n3/n3.0105.fit")[0]

spec1 = hdu1.data
spec2 = hdu2.data

print(hdu1.header['OBJECT'], hdu2.header['OBJECT'])

spec_1ds = []
for spec in [spec1, spec2]:
    plt.figure()
    for i in np.linspace(spec.shape[0]-1, 32).astype(int):
        plt.plot(spec[i,:], marker='', alpha=0.25)

    _idx = spec[int(spec.shape[0]/2)].argmax()
    plt.xlim(_idx-8, _idx+8)
    
    spec_1ds.append(np.mean(spec[:,_idx-3:_idx+3+1], axis=1))

sub_specs = []
for spec_1d in spec_1ds:
    sub_spec = spec_1d[halpha_idx-16:halpha_idx+20]
#     sub_spec = (sub_spec-sub_spec.min()) / (sub_spec.max()-sub_spec.min())
    sub_spec = sub_spec/sub_spec.max()
    sub_spec = sub_spec-np.median(sub_spec)
    sub_specs.append(sub_spec)

plt.plot(sub_specs[0], marker='', drawstyle='steps-mid')
plt.plot(sub_specs[1], marker='', drawstyle='steps-mid')

from scipy.optimize import minimize, leastsq

def line_model(pars, pixel):
    line_ln_amp, line_loc, line_ln_gamma, line_ln_var,*coeff = pars
    v = Voigt1D(line_loc, 
                amplitude_L=-np.exp(line_ln_amp), 
                fwhm_L=np.exp(line_ln_gamma), 
                fwhm_G=np.exp(line_ln_var))
    poly = np.poly1d(coeff)
    return v(pixel) + poly(pixel)

pix_grid = np.arange(len(sub_specs[0]))
p0 = [-0.75, 19., 
       np.log(10.), np.log(4.),
       0., 0.]

plt.plot(pix_grid, sub_specs[0], marker='', drawstyle='steps-mid', zorder=-10)
plt.plot(pix_grid, line_model(p0, pix_grid), marker='', alpha=0.5)

# First just try optimizing the likelihood
p_opts = []

for spec in sub_specs:
    p_opt,ier = leastsq(lambda p: (line_model(p, pix_grid)-spec) / spec_err, x0=p0)
    print(ier)
    p_opts.append(p_opt)

_grid = np.linspace(pix_grid.min(), pix_grid.max(), 256)

fig,axes = plt.subplots(1, 2, sharex=True, figsize=(12,5))

for i,spec,p_opt in zip(range(len(sub_specs)), sub_specs, p_opts):
    axes[i].plot(spec, marker='', drawstyle='steps-mid', zorder=-10)
    axes[i].plot(_grid, line_model(p0, _grid), marker='', alpha=0.5)
    axes[i].plot(_grid, line_model(p_opt, _grid), marker='', alpha=0.5)

d_pix = p_opts[0][1]-p_opts[1][1]
d_pix

d_wvln = pix2wvln(halpha_idx + d_pix) - pix2wvln(halpha_idx)
(d_wvln/6563. * c).to(u.km/u.s)

import emcee
import george
from george import kernels

def model(pars, pixel):
    _, _, line_ln_amp, line_loc, line_ln_gamma, line_ln_var = pars[:6]
    v = Voigt1D(line_loc, 
                amplitude_L=-np.exp(line_ln_amp), 
                fwhm_L=np.exp(line_ln_gamma), 
                fwhm_G=np.exp(line_ln_var))
    return v(pixel)

def ln_prior(pars):
    ln_a, ln_tau, line_ln_amp, line_loc, line_ln_gamma, line_ln_var = pars

    if line_ln_amp < -6 or line_ln_amp > 3:
        return -np.inf
    
    if line_loc < 0 or line_loc > 40:
        return -np.inf
    
    if line_ln_gamma < -10 or line_ln_gamma > 10:
        return -np.inf
    
    if line_ln_var < -10 or line_ln_var > 10:
        return -np.inf
    
    return 0.

def ln_likelihood(pars, pixel, count, count_err):
    a, tau = np.exp(pars[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(pixel, count_err)
    return gp.lnlikelihood(count - model(pars, pixel))

def ln_posterior(pars, pixel, count, count_err):
    try:
        ll = ln_likelihood(pars, pixel, count, count_err)
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf
    
    if not np.any(np.isfinite(ll)):
        return -np.inf
    
    return ln_prior(pars) + ll.sum()

pix_grid = np.arange(len(sub_specs[0]))

p0 = [1E-1, 1E-1, 
       -0.75, 19., 
       np.log(10.), np.log(4.)]

plt.plot(sub_specs[0], marker='', drawstyle='steps-mid', zorder=-10)
plt.plot(pix_grid, model(p0, pix_grid), marker='', alpha=0.5)

spec_err = 0.01 # COMPLETELEY MADE UP
spec = sub_specs[1]

n_walkers = 64
n_dim = len(p0)
all_p0 = emcee.utils.sample_ball(p0, std=1E-4*np.array(p0), size=n_walkers)

sampler = emcee.EnsembleSampler(n_walkers, dim=n_dim, lnpostfn=ln_posterior, args=(pix_grid, spec, spec_err))
pos,prob,_ = sampler.run_mcmc(all_p0, 256)
p = pos[prob.argmax()]
sampler.reset()

# Re-sample the walkers near the best walker from the previous burn-in.
pos = [p + 1e-8 * np.random.randn(n_dim) for i in range(n_walkers)]

print("Running second burn-in...")
pos, prob, _ = sampler.run_mcmc(pos, 256)
p = pos[prob.argmax()]
sampler.reset()

# Re-sample the walkers near the best walker from the previous burn-in.
pos = [p + 1e-8 * np.random.randn(n_dim) for i in range(n_walkers)]

print("Running third burn-in...")
pos, prob, _ = sampler.run_mcmc(pos, 256)
sampler.reset()

print("Running production...")
_ = sampler.run_mcmc(pos, 512)

for dim in range(sampler.chain.shape[-1]):
    plt.figure()
    for wlk in range(sampler.chain.shape[0]):
        plt.plot(sampler.chain[wlk,:,dim], marker='', drawstyle='steps-mid', alpha=0.25, color='k')

plt.plot(spec, drawstyle='steps-mid', marker='')
pixels = np.linspace(pix_grid.min(), pix_grid.max(), 1024)

# for i in range(n_walkers):
for i in range(8):
    pp = sampler.chain[i,-1,:]
#     plt.plot(pix_grid, model(pp, pix_grid), alpha=0.1, marker='')
    
    # Set up the GP for this sample.
    a, tau = np.exp(pp[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(pix_grid, spec_err)

    # Compute the prediction conditioned on the observations and plot it.
    m = gp.sample_conditional(spec - model(pp, pix_grid), pixels) + model(pp, pixels)
    plt.plot(pixels, m, color="#4682b4", alpha=0.3, marker='')

# chain0 = sampler.chain.copy()
# chain1 = sampler.chain.copy()

_,bins,_ = plt.hist(chain0[:,-100:,3].ravel(), bins=np.linspace(9, 12, 64))
plt.hist(chain1[:,-100:,3].ravel(), bins=bins);

d_pix = np.median(chain0[:,-100:,3].ravel()) - np.median(chain1[:,-100:,3].ravel())

d_wvln = pix2wvln(halpha_idx + d_pix) - pix2wvln(halpha_idx)
(d_wvln/6563. * c).to(u.km/u.s)

d_pix

np.median(np.abs(chain0[:,-100:,3].ravel() - np.median(chain0[:,-100:,3].ravel())))

vspan = plt.axvspan(0, 1.1, alpha=0.5, color='g')
vspan.get_xy()

from PyQt5 import QtWidgets

button = QtWidgets.QPushButton()



