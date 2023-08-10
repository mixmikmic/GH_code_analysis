get_ipython().magic('matplotlib inline')

import numpy as np  # for array manipulation
import pylab as pl  # for plotting

logSize = np.arange(2.,4.,0.1)  # kpc

z = 0.01

from astropy.cosmology import WMAP9 as cosmo
D_A = cosmo.kpc_proper_per_arcmin(z)

AngSize = 10**(logSize)/D_A  # arcmin

Area = 2.*(np.pi*(AngSize/4.)**2)  # arcmin^2

mpc2m = 3.09e22 
D_L = cosmo.luminosity_distance(z)*mpc2m

thresh = 2.5e-3  # Jy/beam

bm2amin = 1.13*np.pi*(0.75**2)
print bm2amin

F = (thresh/bm2amin)*Area  # Jy

Prad = 1e-26*F*4*np.pi*D_L**2   # W/Hz/m^2

pl.subplot(111)
pl.plot(10**logSize,Prad)
pl.axis([200.,6000.,1e23,1e29])
pl.loglog()
pl.title(r'Detection threshold for $z=0.01$')
pl.ylabel(r'Radio Power [W/Hz]')
pl.xlabel(r'Size [kpc]')
pl.show()



