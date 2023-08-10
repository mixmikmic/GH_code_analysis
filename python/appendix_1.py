get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt

from astropy.constants import R_sun, au
import astropy.units as u

from methods import integrand_background, G_tot

r_sun_AU = float(R_sun / au)  # Radius of the Sun in AU

def r_to_eps(r, r_obs):
    '''
    Returns Sun - Earth - blob angle assuming
    blob lies on the Thompson sphere.
    
    r is sun-blob distance
    r_obs is sun-observer distance
    '''
    return np.arcsin(r / r_obs)

def f(r, r_obs):
    eps = r_to_eps(r, r_obs)
    # Numerically evaluate the background intensity
    I0 = integrate.quad(integrand_background, 0, np.inf, args=(eps, r_obs))[0]
    # Take a 1/r^2 background number density
    ne = 1 / r**2
    # At large distances, this is the approximate form of the G function
    G = (r / r_sun_AU)**-2
    return G * ne / I0

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
rs = np.logspace(np.log10(0.05), np.log10(0.5), 30)
fs_1 = [f(r, 1) for r in rs]

fs_1 = [f(r, 1) for r in rs]
ax.scatter(rs, fs_1, label='Observer at 1 AU', marker='+')

fs_point3 = [f(r, 0.3) for r in rs[rs < 0.3]]
ax.scatter(rs[rs < 0.3], fs_point3, label='Observer at 0.3 AU', marker='+')

# Fit a line of the form f(r) = a / r
def fit(r, a):
    return a / r

popt, pcov = opt.curve_fit(fit, rs, fs_1, p0=(0.66))
print('Fitted slope: {}'.format(popt[0]))
ax.plot(rs, fit(rs, popt[0]), label='Fit', linestyle='-.')

ax.legend()
ax.set_xlabel('$r_{0}$ (AU)')
ax.set_ylabel('f(r) (1/AU)');

