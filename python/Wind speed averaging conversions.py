get_ipython().magic('matplotlib inline')
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
sns.set_context("poster")

def crossingRate(tau, eta=10.):
    Tu = 3.13 * np.power(eta, 0.2)
    nu = (0.007 + 0.213 * np.power(Tu / tau, 0.654)) / Tu
    return nu

def gustFactor(tau, turbulence, reftime=3600., elev=10.):
    nu = crossingRate(tau, elev)
    Tu = 3.13 * np.power(elev, 0.2)
    sdratio = 1. - 0.193 * np.power((Tu / tau) + 0.1, -0.68)
    v_tau = np.sqrt(2. * np.log(reftime * nu)) +         0.577 / np.sqrt(2. * np.log(reftime * nu))
    g = v_tau * sdratio
    return 1. + g * turbulence

def esdu_gto(reftime):
    gto = 0.2193 * np.log(np.log10(reftime)) + 0.7242
    return gto

reftime = 60.
tau = np.array([0.1, 0.2, .5, 1., 3., 10., 30., 60., 120., 180., 600., 3600.])
eta = 10.

gto = esdu_gto(reftime)

gf_inland = gustFactor(tau, 0.25, reftime) * gto
gf_offland = gustFactor(tau, 0.2, reftime) * gto
gf_offsea = gustFactor(tau, 0.15, reftime) * gto
gf_atsea = gustFactor(tau, 0.1, reftime) * gto

gf_inland = np.where(gf_inland < 1., 1., gf_inland)
gf_offland = np.where(gf_offland < 1., 1., gf_offland)
gf_offsea = np.where(gf_offsea < 1., 1., gf_offsea)
gf_atsea = np.where(gf_atsea < 1., 1., gf_atsea)

plt.semilogx(tau, gf_inland, label=r"In-land ($I_u=0.250$)")
plt.semilogx(tau, gf_offland, label=r"Off-land ($I_u=0.200$)")
plt.semilogx(tau, gf_offsea, label=r"Off-sea ($I_u=0.150$)")
plt.semilogx(tau, gf_atsea, label=r"At-sea ($I_u=0.100$)")
plt.title("Gust factors ($T_o = {:.0f}$ seconds)".format(reftime))
plt.legend(frameon=True)
plt.ylim((1.0, 2.0))
plt.xlim((0.1, 1000))
plt.ylabel(r'Gust factor $G_{\tau, T_o}$')
plt.xlabel(r'Gust duration $\tau$ (s)')
plt.grid(which='major')
plt.grid(which='minor', linestyle='--', linewidth=1)
plt.legend()
sns.despine()

titlestr = "Duration| " + "| ".join(['{:6.1f}']*len(tau)).format(*tau)
print titlestr

rowfmt = "{:s}| " + "| ".join(['{:6.3f}']*len(tau))
print rowfmt.format("In-land ", *gf_inland)
print rowfmt.format("Off-land", *gf_offland)
print rowfmt.format("Off-sea ", *gf_offsea)
print rowfmt.format("At-sea  ", *gf_atsea)



