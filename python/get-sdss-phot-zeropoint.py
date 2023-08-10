from astropy.io import fits
import numpy as np

image = 'frame-r-003893-4-0293.fits.bz2'

dat = fits.open(image)

dat.info()

dat[0].header

flux_zp_nmgy = dat[0].header['NMGY']
print zp_nmgy

exptime = float(dat[0].header['EXPTIME'])
print exptime

zp = 22.5 - 2.5*np.log10(exptime)
print zp



