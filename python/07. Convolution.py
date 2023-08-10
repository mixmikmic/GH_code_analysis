import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from astropy.io import fits
from astropy.convolution import convolve, convolve_fft

hdu = fits.open('../data/3C_273-S-B-bcc2009.fits')[0]
flux = hdu.data / 10000

a = hdu.header['CRVAL1']
b = hdu.header['CRVAL1'] + hdu.header['CDELT1']*len(flux)

x = np.linspace(a, b, len(flux))

box_kernel = Box1DKernel(3)

plt.plot(box_kernel, drawstyle='steps')
plt.xlim(-1, 3)
plt.xlabel('x [pixels]')
plt.ylabel('value')
plt.show()

smoothed_data_box = convolve(flux, box_kernel)

plt.plot(x,flux)
plt.plot(x,smoothed_data_box)
plt.legend(['original','box'],loc=2)

gauss_kernel = Gaussian1DKernel(5)
smoothed_data_gauss = convolve(flux, gauss_kernel)

plt.plot(x,flux)
plt.plot(x,smoothed_data_gauss)
plt.legend(['original','gauss'],loc=2)

data = fits.getdata('../data/MESSIER_051-I-103aE-dss1.fits')

plt.imshow(data, cmap='gray')
plt.colorbar()

from astropy.convolution import Gaussian2DKernel

gauss = Gaussian2DKernel(stddev=2)
cdata = convolve(data, gauss, boundary='extend')

plt.imshow(cdata, cmap='gray',)
plt.colorbar()

