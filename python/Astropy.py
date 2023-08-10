from __future__ import print_function, division # Do this for Python 2/3 compatibility

import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

import astropy.units as u

d1 = np.random.random(5) * u.m    # u.meter also works
d2 = np.random.random(5) * u.pc   # u.parsec also works

print(d1)
print(d2)

print(d2/d1)

print((d2/d1).decompose())

theta1 = 1.7 * u.deg
theta2 = 25. * u.arcsec

print((theta1/theta2).decompose())

print('{:.5g} = {:.5g} = {:.5g}'.format(theta1, theta1.to(u.rad), theta1.to(u.arcmin)))

print(d2.si)
print(d2.cgs)

import astropy.constants as const

print(const.G)

print(const.G.uncertainty)

r_s = const.G * const.M_sun / const.c**2

print('r_s = {:.3g} = {:.3g}'.format(r_s.to(u.km), r_s.to(u.AU)))

from astropy.coordinates import SkyCoord

c1 = SkyCoord(100., 45., frame='galactic', unit='deg')
c2 = SkyCoord(60.*u.deg, -25.*u.deg, frame='galactic')
c3 = SkyCoord('10h50m35.3s', '+12d17m1.0s', frame='icrs')
c3 = SkyCoord('10:50:35.3', '+12:17:1.0', frame='icrs', unit=(u.hourangle, u.deg))

ra = 360. * np.random.random(5)
dec = 180. * np.random.random(5) - 90.

c4 = SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')

print(c4)

c4_gal = c4.transform_to('galactic')
c4_icrs = c4.transform_to('icrs')

print(c4_gal)
print(c4_icrs)

print(c4_icrs.ra)
print(c4_icrs.ra.value)
print(c4_icrs.ra.unit)
print('')
print(c4_icrs.dec)
print(c4_icrs.dec.value)
print(c4_icrs.dec.unit)

c5 = SkyCoord(45.*u.deg, 10.*u.deg, distance=1.*u.kpc, frame='galactic')
print(c5)
print('d = {}'.format(c5.distance))
print('x, y, z = {}'.format(c5.cartesian))

from astropy.coordinates import EarthLocation
EarthLocation.of_site('Cerro Tololo Interamerican Observatory')

from astropy.io import ascii

data = ascii.read('ascii1.txt')
print(data)

data = ascii.read('ascii2.txt')
print(data)

print(data['obsid'][1])

data = ascii.read('ascii3.txt')

data = ascii.read('ascii3.txt', format='fixed_width', data_start=2)
print(data)

import astropy.io.fits as fits

fname = 'dss17460a27m.fits'
hdulist = fits.open(fname)

hdulist.info()

hdulist[0].data

plt.imshow(hdulist[0].data, cmap='binary', interpolation='nearest');

hdulist.close()

dtype = [
    ('x', 'f8'),
    ('y', 'f8'),
    ('velocity', 'f8'),
    ('objid', 'i4'),
    ('objname', 'S10')  # A string with a maximum length of 10 characters
]

data = np.empty(5, dtype=dtype)
data['x'][:] = np.random.random(5)
data['y'][:] = np.random.random(5)
data['velocity'][:] = np.random.random(5)
data['objid'][:] = np.arange(5)
data['objname'][:] = 'a b c d e'.split()

hdu = fits.BinTableHDU(data=data)

hdulist = fits.HDUList([fits.PrimaryHDU(), hdu]) # We need a dummy primary HDU
hdulist.writeto('my_table.fits', clobber=True)  # clobber=True means that any existing file will be overwritten
hdulist.close()

import astropy.wcs as wcs

fname = 'dss17460a27m.fits'
hdulist = fits.open(fname)

w = wcs.WCS(hdulist[0].header)
hdulist.close()

pix_coords = np.array([
    [10., 5.],
    [3., 8.],
    [20.5, 10.1]
])

world_coords = w.wcs_pix2world(pix_coords, 1)

print(world_coords)

world_coords = np.array([
    [92.356, 20.4775],
    [92.353, 20.4779]
])

pix_coords = w.wcs_world2pix(world_coords, 1)

print(pix_coords)

import astropy.convolution as conv

hdulist = fits.open(fname)
img_orig = hdulist[0].data[:]
hdulist.close()

kern = conv.Gaussian2DKernel(5.)

img_smooth = conv.convolve(img_orig, kern, boundary='extend')

fig = plt.figure()

ax = fig.add_subplot(1,2,1)
ax.imshow(img_orig, cmap='binary', interpolation='nearest')
ax.axis('off')

ax = fig.add_subplot(1,2,2)
ax.imshow(img_smooth, cmap='binary', interpolation='nearest')
ax.axis('off');

img_orig[5:10, 5:10] = np.nan
img_smooth = conv.convolve(img_orig, kern, boundary='extend')

fig = plt.figure()

ax = fig.add_subplot(1,2,1)
ax.imshow(img_orig, cmap='binary', interpolation='nearest')
ax.axis('off')

ax = fig.add_subplot(1,2,2)
ax.imshow(img_smooth, cmap='binary', interpolation='nearest')
ax.axis('off')



