from astropy.nddata import CCDData
from photutils import detect_threshold, detect_sources, source_properties, properties_table
from matplotlib import pyplot as plt
img = CCDData.read('extented_tutorial/Leo.fits', unit='adu')

thresh = detect_threshold(data=img.data, snr=3)
# This will give you 3*bkg_std.

segm = detect_sources(data=img.data, threshold=thresh, npixels=100, connectivity=8)

fig, ax = plt.subplots(1,2, figsize=(15,15))
ax[0].imshow(img.data, vmin=220, vmax=400, origin='lower')
ax[0].imshow(segm, alpha=0.2, origin='lower')
ax[1].imshow(segm, origin='lower')
plt.show()

source_props = source_properties(img.data, segm, background=thresh/3)
# The "background" parameter should be inserted as the sky background value.
# All the source properties are calculated after this background subtraction.
proptable = properties_table(source_props)
proptable.pprint(max_width=100)

import numpy as np
from photutils import Background2D
from photutils.utils import filter_data
from astropy.convolution import Gaussian2DKernel

bkg = Background2D(img.data, box_size=(32,32), filter_size=(3,3))
# Background estimation with (32x32) mesh, 3x3 median filter.
# By default, SExtractorBackground() is used.

thresh = bkg.background + 3 * bkg.background_rms
# thresh = bkg + 3 * bkg_std

FWHM = 3.
sigma = FWHM / (2*np.sqrt(2*np.log(2))) # FWHM to sigma

# make kernel and normalize
kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
kernel.normalize()

# find sources using convolution
segm = detect_sources(img.data, thresh, npixels=100, filter_kernel=kernel)

fig, ax = plt.subplots(1,2, figsize=(15,15))
ax[0].imshow(img.data, vmin=220, vmax=400, origin='lower')
ax[0].imshow(segm, alpha=0.2, origin='lower')
ax[1].imshow(segm, origin='lower')
plt.show()

convolved = filter_data(img.data, kernel=kernel)

fig, ax = plt.subplots(1,2, figsize=(15,15))
ax[0].imshow(convolved, vmin=220, vmax=400, origin='lower')
ax[0].set_title('Gaussian convolved image')
ax[1].imshow(img.data/convolved, origin='lower', vmin=0, vmax=2)
ax[1].set_title('original / convolved')
plt.show()

gain = img.header['egain'] # around 2
ronoise = 10               # arbitrary
error = np.sqrt(img.data/gain + (ronoise/gain)**2 + bkg.background_rms**2)
# errormap = sqrt(Poisson + ronoise**2 + bkg_std**2)

source_props = source_properties(img.data, segm, background=bkg.background, error=error)
# The "background" parameter should be inserted as the sky background value.
# All the source properties are calculated after this background subtraction.
proptable = properties_table(source_props)
proptable.pprint(max_width=100)

proptable.sort('area')
prop_galaxy = proptable[-3:]
prop_galaxy.pprint(max_width=100)

from ccdproc import CCDData
from astropy.convolution import Gaussian2DKernel
from astropy.wcs import WCS
from photutils import detect_sources, source_properties, properties_table

img = CCDData.read('extended_tutorial/ngc1132_r.fits', hdu=0, unit='u.electron/u.s')
# sky has already been subtracted, so essentially 0
sky_std = 0.20
thresh = 3 * sky_std
FWHM = 1.37
# https://dr12.sdss.org/fields/name?name=ngc1132

sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
kernel.normalize()
wcs = WCS(img.header)  
# If you put WCS to `source_properties`, the centroid RA/DEC will also be calculated.

segm = detect_sources(img.data, 
                      threshold=thresh, 
                      npixels=500,      # Larger npixel to detect only few sources
                      connectivity=8, 
                      filter_kernel=kernel)
source_prop = source_properties(img.data, segm, wcs=wcs)
proptable = properties_table(source_prop)
proptable.sort('ellipticity')

# Choose the one with the largest ellipticity
ngc1132 = proptable[-1] 
print(ngc1132.colnames)
# print out all the parameters calculated from `source_properties`.

from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

ra_sdss, de_sdss = 43.2159167, -1.2747222
radec_sdss = SkyCoord(ra_sdss, de_sdss, unit="deg")
x_sdss, y_sdss = radec_sdss.to_pixel(wcs)
x_cent, y_cent = (ngc1132['xcentroid'], ngc1132['ycentroid'])


#Plot
fig, ax = plt.subplots(1,1, figsize=(12,12))

ax.imshow(img.data, vmin=0, vmax=1, origin='lower')
ax.plot(x_sdss, y_sdss, marker='+', ms=10, color='r')
ax.plot(x_cent, y_cent, marker='x', ms=10, color='b')

# Make zoom-in
ax2 = zoomed_inset_axes(ax, 6, loc=9) # zoom-factor: 2.5, location: upper-left
ax2.imshow(img.data, vmin=0, vmax=1, origin='lower')
ax2.plot(x_sdss, y_sdss, marker='+', ms=10, color='r')
ax2.plot(x_cent, y_cent, marker='x', ms=10, color='b')
halfcb = 50
ax2.set_xlim(x_cent-halfcb, x_cent+halfcb) # apply the x-limits
ax2.set_ylim(y_cent-halfcb, y_cent+halfcb) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, ax2, loc1=3, loc2=1, fc="none", ec="0.5")

plt.show()

from photutils import EllipticalAnnulus as EllAn
from photutils import aperture_photometry as APPHOT
e = ngc1132['elongation'] # := a/b
theta = ngc1132['orientation']
nap = 100
annul = [EllAn(positions=(x_cent, y_cent), a_in=b0*e, a_out=(b0+1)*e, 
               b_out=(b0+1), theta=theta) for b0 in range(1, nap+1)]
phot = APPHOT(img.data, annul)

#Plot
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.imshow(img.data, vmin=0, vmax=1, origin='lower')
[annul[i].plot(ax=ax, color='red') for i in (10, 30, 50, 70)]
# Make zoom-in
ax2 = zoomed_inset_axes(ax, 6, loc=9) # zoom-factor: 2.5, location: upper-left
ax2.imshow(img.data, vmin=0, vmax=1, origin='lower')
ax2.plot(x_sdss, y_sdss, marker='+', ms=10, color='r')
ax2.plot(x_cent, y_cent, marker='x', ms=10, color='b')
halfcb = 50
ax2.set_xlim(x_cent-halfcb, x_cent+halfcb) # apply the x-limits
ax2.set_ylim(y_cent-halfcb, y_cent+halfcb) # apply the y-limits
[annul[i].plot(ax=ax2, color='red') for i in (10, 30, 50, 70)]
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, ax2, loc1=3, loc2=1, fc="none", ec="0.5")

plt.show()

semimaj = np.arange(1, nap+1)*e
counts = np.zeros(nap)
dcounts = np.zeros(nap)
B_surf = []
for i in range(0, nap):
    count = phot[phot.colnames[i+3]]
    counts[i] = count
    # phot.colnames = column names = "aperture_sum_X"
    dcount = count / annul[i].area() # count per pixel
    dcounts[i] = dcount
    bright = np.log10(dcount)
    B_surf.append(bright)

plt.plot(semimaj, dcounts, ls=':', marker='x')
plt.xscale('log')
plt.ylabel('intensity (e/s/pixel)')
plt.xlabel('log semimajor axis (pixel)')
plt.grid(ls=":")
plt.show()

from astropy.modeling.functional_models import Sersic1D
from astropy.modeling.fitting import LevMarLSQFitter

f_init = Sersic1D(amplitude=ngc1132['max_value'], r_eff=10, n=4.)
fitter = LevMarLSQFitter()
fitted = fitter(f_init, semimaj, dcounts)
print(fitted)

plt.plot(semimaj, dcounts, ls=':', marker='x', label='Observed intensity')
plt.plot(semimaj, fitted(semimaj), ls='-', 
         label='Fit($A={:.3f}$, $r_e={:.1f}$, $n={:.2f}$)'.format(fitted.amplitude.value,
                                                   fitted.r_eff.value,
                                                   fitted.n.value))
plt.xscale('log')
plt.grid(ls=':')
plt.ylabel('intensity (e/s/pixel)')
plt.xlabel('log semimajor axis (pixel)')
plt.legend()
plt.show()



