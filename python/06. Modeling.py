import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from astropy.io import fits
from astropy.modeling import models, fitting

hdu = fits.open('../data/3C_273-S-B-bcc2009.fits')[0]
flux = hdu.data / 10000

a = hdu.header['CRVAL1']
b = hdu.header['CRVAL1'] + hdu.header['CDELT1']*len(flux)

a,b

x = np.linspace(a, b, len(flux))
x

plt.plot(x,flux)

fitter = fitting.LevMarLSQFitter()

m = models.Const1D() + models.Gaussian1D(mean=6500)
m_new = fitter(m, x, flux)

plt.plot(x, flux)
plt.plot(x, m_new(x))

p1 = models.PowerLaw1D(amplitude=1.5, x_0=5500, alpha=2)
m_new = fitter(p1,x,flux)
plt.plot(x, flux)
plt.plot(x, m_new(x))

g1 = models.Gaussian1D(mean=6450)
g2 = models.Gaussian1D(mean=4520)
g3 = models.Gaussian1D(mean=3950)
m = p1 + g1 + g2 + g3 
m_new = fitter(m,x,flux)
plt.plot(x, flux)
plt.plot(x, m_new(x))



