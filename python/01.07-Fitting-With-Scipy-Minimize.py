from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import G
from math import pi
get_ipython().run_line_magic('matplotlib', 'inline')

h = fits.open('data/kplr006922244-2010078095331_llc.fits')
h[1].data.names
t,flux,err=h[1].data['TIME'],h[1].data['PDCSAP_FLUX'],h[1].data['PDCSAP_FLUX_ERR']
t,flux,err=t[np.isfinite(flux)],flux[np.isfinite(flux)],err[np.isfinite(err)]

per=3.52233911024
phase = t  % per

plt.figure(figsize=(15,3))
plt.scatter(phase,flux,s=0.5)
plt.xlabel('Phase')
plt.ylabel('Counts (e$^-$/s)')

r = h[0].header['RADIUS']*u.solRad
logg = h[0].header['LOGG']
feh = h[0].header['FEH']
teff = h[0].header['TEFF']*u.K

g = 10.**(logg)*(u.cm/u.second**2)
m = ((g*r**2)/G).to(u.solMass)

rp_init=((np.median(flux)-np.min(flux))/np.median(flux))**0.5
i_init=90

t0_init=1.30
d=(((per*u.day).to(u.second))**2*G*m/(4*pi**2))**(1./3.)
a_init=(d.to(u.solRad)/r).value

import batman
params = batman.TransitParams()       #object to store transit parameters
params.ecc = 0.                       #eccentricity
params.w = 90.                        #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.5230, 0.1218]           #limb darkening coefficients
params.per = per                      #orbital period



def func(guess,return_model=False):
    params.rp=guess[0]
    params.a=guess[1]
    params.inc=guess[2]
    params.t0=guess[3]
    
    m = batman.TransitModel(params, np.sort(phase))
    model=m.light_curve(params)*np.median(flux)
    
    if return_model:
        return(model)
    else:
        return((np.nansum((flux[np.argsort(phase)]-model)**2/flux[np.argsort(phase)])))

from scipy.optimize import minimize
res=minimize(func,[rp_init,a_init,i_init,t0_init],method='Powell')

lcm = func([rp_init,a_init,i_init,t0_init,0.5230,0.1218],return_model=True)
lcm_fit = func(res.x,return_model=True)

#print (rp_init,a_init,i_init,t0_init)
print(res.x[0:3])
planet_params=[0.097659, 6.84062, 83.97799]
print(planet_params)

fig,ax=plt.subplots(2,figsize=(15,5))
ax[0].errorbar(phase,flux,err,ls='',label='Data')
ax[0].set_ylabel('Counts (e$^-$/s)')
ax[0].plot(np.sort(phase),lcm,c='C1',label='Initial Guess',lw=3)
ax[0].plot(np.sort(phase),lcm_fit,c='C3',label='Optimal Model',lw=3)
ax[0].legend()
ax[0].set_xticks([])
ax[0].set_title('Folded Transit and Best Fit Model')

ax[1].errorbar(np.sort(phase),flux[np.argsort(phase)]-lcm_fit,err,ls='',label='Data')
ax[1].set_xlabel('Phase Folded Transit (Days)')
ax[1].set_ylabel('Counts')
ax[1].set_title('Residuals')

ax[0].set_xlim(1.18,1.57)
ax[1].set_xlim(1.18,1.57)

