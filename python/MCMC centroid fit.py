from os import path

from astropy.io import fits
from astropy.constants import c
import astropy.coordinates as coord
from astropy.table import Table
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')
from scipy.stats import scoreatpercentile

import emcee
import corner

from comoving_rv.longslit import extract_region
from comoving_rv.longslit.fitting import VoigtLineFitter
from comoving_rv.longslit.models import voigt_polynomial, binned_voigt_polynomial
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo,
                                  SpectralLineInfo, SpectralLineMeasurement)
from comoving_rv.plot import colors

base_path = '/Volumes/ProjectData/gaia-comoving-followup/'
db_path = path.join(base_path, 'db.sqlite')
engine = db_connect(db_path)
session = Session()

np.random.seed(123)
id_ = np.random.choice([x[0] for x in session.query(Observation.id).all()])

Halpha = session.query(SpectralLineInfo)                .filter(SpectralLineInfo.name == 'Halpha').one()

obs = session.query(Observation).filter(Observation.id == id_).one()

# Read the spectrum data and get wavelength solution
filename_1d = obs.path_1d(base_path)
spec = Table.read(filename_1d)

# Extract region around Halpha
x, (flux, ivar) = extract_region(spec['wavelength'],
                                 center=Halpha.wavelength.value,
                                 width=128,
                                 arrs=[spec['source_flux'],
                                       spec['source_ivar']])

plt.plot(x, flux, marker='', drawstyle='steps-mid')

absorp_emiss = -1. # assume absorption
lf = VoigtLineFitter(x, flux, ivar, absorp_emiss=absorp_emiss)
lf.fit()
fit_pars = lf.get_gp_mean_pars()

lf.gp.get_parameter_names()

param_bounds = [(-8, 14), (-8, 14), 
                (2, 16), (6547, 6579),
                (-4, 2), (-4, 2), 
                (0, 1E16), (-np.inf, np.inf)]

lf.gp.kernel.parameter_bounds = [(None,None)] * 2
lf.gp.mean.parameter_bounds = [(None,None)] * 6

def ln_posterior(pars, gp, flux_data):
    gp.set_parameter_vector(pars)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf

    # HACK: Gaussian prior on log(rho)
    var = 1.
    lp += -0.5*(pars[1]-1)**2/var - 0.5*np.log(2*np.pi*var)
    
    for i, par, bounds in zip(range(len(pars)), pars, param_bounds):
        if par < bounds[0] or par > bounds[1]:
            return -np.inf

    ll = gp.log_likelihood(flux_data)
    if not np.isfinite(ll):
        return -np.inf

    return ll + lp

initial = np.array(lf.gp.get_parameter_vector())
ndim, nwalkers = len(initial), 64

p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=(lf.gp, flux))

print("Running burn-in...")
p0, lp, _ = sampler.run_mcmc(p0, 128)
print("Running 2nd burn-in...")
sampler.reset()
p0 = p0[lp.argmax()] + 1e-3 * np.random.randn(nwalkers, ndim)
p0, lp, _ = sampler.run_mcmc(p0, 512)
print("Running production...")
sampler.reset()
pos, lp, _ = sampler.run_mcmc(p0, 4096)

fit_kw = dict()
for i,par_name in enumerate(lf.gp.get_parameter_names()):
    if 'kernel' in par_name: continue

    # remove 'mean:'
    par_name = par_name[5:]

    # skip bg
    if par_name.startswith('bg'): continue

    samples = sampler.flatchain[:,i]

    if par_name.startswith('ln_'):
        par_name = par_name[3:]
        samples = np.exp(samples)

    MAD = np.median(np.abs(samples - np.median(samples)))
    fit_kw[par_name] = np.median(samples)
    fit_kw[par_name+'_error'] = 1.5 * MAD # convert to ~stddev

fit_kw

par_name_map = dict()
par_name_map['kernel:log_sigma'] = r'$\ln\sigma_{3/2}$'
par_name_map['kernel:log_rho'] = r'$\ln\rho_{3/2}$'
par_name_map['mean:ln_amp'] = r'$\ln A$'
par_name_map['mean:x0'] = r'$x_0 - \lambda_{{\rm H}\alpha}$'
par_name_map['mean:ln_std_G'] = r'$\ln\sigma_G$'
par_name_map['mean:ln_hwhm_L'] = r'$\ln\gamma_L$'
par_name_map['mean:bg0'] = r'$\alpha_1$'
par_name_map['mean:bg1'] = r'$\alpha_2$'

lims = dict()
lims['kernel:log_sigma'] = (5, 10)
lims['kernel:log_rho'] = (0, 5)
lims['mean:ln_amp'] = (11.5, 13)
lims['mean:x0'] = (6562.5-Halpha.wavelength.value, 6563.5-Halpha.wavelength.value)
lims['mean:ln_std_G'] = (-4, 2)
lims['mean:ln_hwhm_L'] = (-4, 2)
lims['mean:bg0'] = (6E4, 8E4)
lims['mean:bg1'] = (-200, 0)

# plot MCMC traces
fig,axes = plt.subplots(4, 2, figsize=(6.5,9), sharex=True)
for i in range(sampler.dim):
    long_name = lf.gp.get_parameter_names()[i]
    if long_name == 'mean:x0':
        x = Halpha.wavelength.value
    else:
        x = 0.
    
    axes.flat[i].set_rasterization_zorder(1)
    for walker in sampler.chain[...,i]:
        axes.flat[i].plot(walker[:1024] - x, marker='', drawstyle='steps-mid', 
                          alpha=0.1, color=colors['not_black'], zorder=-1, linewidth=1.)
    axes.flat[i].set_ylabel(par_name_map[long_name], fontsize=18)
    axes.flat[i].set_ylim(lims[long_name])

axes.flat[i].set_xlim(0, 1024)
axes.flat[i].xaxis.set_ticks([0, 512, 1024])

axes[-1,0].set_xlabel('MCMC step num.')
axes[-1,1].set_xlabel('MCMC step num.')
    
fig.tight_layout()

# HACK: this one happens to have an HD number
fig.suptitle('Source: HD {}'.format(obs.simbad_info.hd_id), y=0.97, fontsize=20)
fig.subplots_adjust(top=0.92)

fig.savefig('mcmc_trace.pdf', dpi=300)

flatchain = np.vstack(sampler.chain[:,::16])

def mean_line_only(x, line_fitter):
    mn = line_fitter.gp.mean
    v = binned_voigt_polynomial(x, mn._absorp_emiss*np.exp(mn.ln_amp),
                                mn.x0, np.exp(mn.ln_std_G), np.exp(mn.ln_hwhm_L),
                                [0. for i in range(mn._n_bg_coef)])
    return v

def smooth_mean_line_only(x, line_fitter):
    mn = line_fitter.gp.mean
    v = voigt_polynomial(x, mn._absorp_emiss*np.exp(mn.ln_amp),
                         mn.x0, np.exp(mn.ln_std_G), np.exp(mn.ln_hwhm_L),
                         [0. for i in range(mn._n_bg_coef)])
    return v

def poly_only(x, line_fitter):
    mn = line_fitter.gp.mean
    v = voigt_polynomial(x, 0,
                         mn.x0, np.exp(mn.ln_std_G), np.exp(mn.ln_hwhm_L),
                         [getattr(mn, "bg{}".format(i)) for i in range(mn._n_bg_coef)])
    return v

def gp_only(x, line_fitter):
    mu, var = lf.gp.predict(lf.flux, x, return_var=True)
    return mu - poly_only(x, line_fitter) - mean_line_only(x, line_fitter), np.sqrt(var)

def bg_only(x, line_fitter):
    mu, var = lf.gp.predict(lf.flux, x, return_var=True)
    return mu - mean_line_only(x, line_fitter)

data_plot_style = dict(color=colors['not_black'], marker='', drawstyle='steps-mid', zorder=-10)
data_errorbar_style = dict(color='#888888', marker='', linestyle='', zorder=-12)

fig,axes = plt.subplots(2, 1, figsize=(5, 6.5), sharex=True)

wave_grid = np.linspace(lf.x.min(), lf.x.max(), 1024)

# data
axes[0].plot(lf.x, lf.flux, **data_plot_style)
axes[0].errorbar(lf.x, lf.flux, 1/np.sqrt(lf.ivar), **data_errorbar_style)

axes[1].plot(lf.x, lf.flux - mean_line_only(lf.x, lf) - poly_only(lf.x, lf), **data_plot_style)
axes[1].errorbar(lf.x, lf.flux - mean_line_only(lf.x, lf) - poly_only(lf.x, lf), 1/np.sqrt(lf.ivar),
                 **data_errorbar_style)

all_fits = np.zeros((len(flatchain[::10]), len(wave_grid)))
for i,pars in enumerate(flatchain[::10]):
    lf.gp.set_parameter_vector(pars)
    all_fits[i] = smooth_mean_line_only(wave_grid, lf) + poly_only(wave_grid, lf)

lo,hi = scoreatpercentile(all_fits, [15,85], axis=0)
axes[0].fill_between(wave_grid, lo, hi, color=colors['fit'], 
                     alpha=0.4, zorder=10, linewidth=0)

pars = np.median(flatchain, axis=0)
lf.gp.set_parameter_vector(pars)
axes[0].plot(wave_grid, smooth_mean_line_only(wave_grid, lf) + poly_only(wave_grid, lf), 
                 color=colors['fit'], marker='', zorder=12, alpha=1.)

# model/data with line removed
lf.gp.set_parameter_vector(pars)
mu, std = gp_only(wave_grid, lf)
axes[1].plot(wave_grid, mu, color=colors['gp_model'], marker='')

axes[1].set_xlim(wave_grid.min(), wave_grid.max())
axes[1].set_xlabel(r'wavelength [${\rm \AA}$]')

axes[0].set_ylabel('flux')
axes[1].set_ylabel('residuals')
# axes[1].set_ylim(-4100, 4100)

axes[0].ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
axes[1].ticklabel_format(style='sci',scilimits=(-3,3),axis='y')

axes[0].xaxis.set_ticks(np.arange(6500, 6625+1, 25))
axes[0].yaxis.set_ticks(np.arange(4, 8.+0.1, 1)*1e4)
axes[1].yaxis.set_ticks([-2.5e3, 0, 2.5e3])

fig.tight_layout()

# HACK: this one happens to have an HD number
fig.suptitle('Source: HD {}'.format(obs.simbad_info.hd_id), y=0.97, fontsize=20)
fig.subplots_adjust(top=0.92)

fig.savefig('mcmc_example_fit.pdf')

_flatchain = flatchain.copy()
_flatchain[:,3] = _flatchain[:,3] - Halpha.wavelength.value

# lims = scoreatpercentile(flatchain, per=[1, 99], axis=0).T.tolist()
tmp = np.median(_flatchain, axis=0)
lims = tmp[None] + np.array([-3, 3])[:,None] * np.std(_flatchain, axis=0)[None]
lims = lims.T.tolist()

# corner plot
fig = corner.corner(_flatchain, range=lims,
                    labels=[par_name_map[x]
                            for x in lf.gp.get_parameter_names()])

# HACK: this one happens to have an HD number
fig.suptitle('Source: HD {}'.format(obs.simbad_info.hd_id), y=0.97, fontsize=32)

for ax in fig.axes:
    ax.set_rasterization_zorder(1)

fig.savefig('mcmc_corner.pdf', dpi=300)



