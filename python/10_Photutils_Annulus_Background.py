import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAnnulus as CircAn

#%%
hdu = fits.open('HST_Tutorial/M13.fits')
img = hdu[0].data[900:1200, 900:1200]
# if image value < 10^(-6), replace the pixel as 10^(-6)
img[img < 1.e-6] = 1.e-6

FWHM   = 2.5
sky_th = 500   # sky_th * sky_sigma will be used for detection lower limit
sky_a, sky_m, sky_s  = sigma_clipped_stats(img) # 3-sigma, 5 iters
thresh = sky_th*sky_s

find   = DAOStarFinder(fwhm=FWHM, threshold=thresh,
                       sharplo=0.2, sharphi=1.0,  # default values 
                       roundlo=-1.0, roundhi=1.0, # default values
                       sigma_radius=1.5,          # default values
                       ratio=1.0,                 # 1.0: circular gaussian
                       exclude_border=True)       # To exclude sources near edges
found = find(img)

# Use the object "found" for aperture photometry:
# save XY coordinates:
coord = (found['xcentroid'], found['ycentroid']) 
annul = CircAn(positions=coord, r_in=4*FWHM, r_out=6*FWHM)

#%%
plt.figure(figsize=(10,10))
plt.imshow(img, vmax=0.10)
annul.plot(color='red', lw=1., alpha=0.7)
plt.colorbar()
plt.show()

# since our `annul` has many elements, let me use [3] to use only the 4th annulus:
mask_annul = (annul.to_mask(method='center'))[3]
# CAUTION!! YOU MUST USE 'center', NOT 'exact'!!!

cutimg = mask_annul.cutout(img)
plt.imshow(cutimg, vmin=0.005, vmax=0.05, origin='lower')
plt.show()

sky_apply  = mask_annul.apply(img)
plt.imshow(sky_apply, origin='lower', vmin=0.005, vmax=0.05)
plt.show()

import numpy as np
from astropy.stats import sigma_clip



def sky_fit(all_sky, method='mode', sky_nsigma=3, sky_iter=5,             mode_option='sex', med_factor=2.5, mean_factor=1.5):
    '''
    Estimate sky from given sky values.

    Parameters
    ----------
    all_sky : ~numpy.ndarray
        The sky values as numpy ndarray format. It MUST be 1-d for proper use.
    method : {"mean", "median", "mode"}, optional
        The method to estimate sky value. You can give options to "mode"
        case; see mode_option.
        "mode" is analogous to Mode Estimator Background of photutils.
    sky_nsigma : float, optinal
        The input parameter for sky sigma clipping.
    sky_iter : float, optinal
        The input parameter for sky sigma clipping.
    mode_option : {"sex", "IRAF", "MMM"}, optional.
        sex  == (med_factor, mean_factor) = (2.5, 1.5)
        IRAF == (med_factor, mean_factor) = (3, 2)
        MMM  == (med_factor, mean_factor) = (3, 2)

    Returns
    -------
    sky : float
        The estimated sky value within the all_sky data, after sigma clipping.
    std : float
        The sample standard deviation of sky value within the all_sky data,
        after sigma clipping.
    nsky : int
        The number of pixels which were used for sky estimation after the
        sigma clipping.
    nrej : int
        The number of pixels which are rejected after sigma clipping.
    -------

    '''
    sky = all_sky.copy()
    if method == 'mean':
        return np.mean(sky), np.std(sky, ddof=1)

    elif method == 'median':
        return np.median(sky), np.std(sky, ddof=1)

    elif method == 'mode':
        sky_clip   = sigma_clip(sky, sigma=sky_nsigma, iters=sky_iter)
        sky_clipped= sky[np.invert(sky_clip.mask)]
        nsky       = np.count_nonzero(sky_clipped)
        mean       = np.mean(sky_clipped)
        med        = np.median(sky_clipped)
        std        = np.std(sky_clipped, ddof=1)
        nrej       = len(all_sky) - len(sky_clipped)

        if nrej < 0:
            raise ValueError('nrej < 0: check the code')

        if nrej > nsky: # rejected > survived
            raise Warning('More than half of the pixels rejected.')

        if mode_option == 'IRAF':
            if (mean < med):
                sky = mean
            else:
                sky = 3 * med - 2 * mean

        elif mode_option == 'MMM':
            sky = 3 * med - 2 * mean

        elif mode_option == 'sex':
            if (mean - med) / std > 0.3:
                sky = med
            else:
                sky = (2.5 * med) - (1.5 * mean)
        else:
            raise ValueError('mode_option not understood')

        return sky, std, nsky, nrej

sky_apply  = mask_annul.apply(img)
sky_non0   = np.nonzero(sky_apply)
sky_pixel  = sky_apply[sky_non0]
msky, stdev, nsky, nrej = sky_fit(sky_pixel, method='mode', mode_option='sex')
print(msky, sky_std, nsky, nrej)

N_stars = len(found)
print('Star ID    msky  s_s     nsky nrej')
for i in range(0, N_stars):
    mask_annul = (annul.to_mask(method='center'))[i]
    sky_apply  = mask_annul.apply(img)
    sky_non0   = np.nonzero(sky_apply)
    sky_pixel  = sky_apply[sky_non0]
    msky, sky_std, nsky, nrej = sky_fit(sky_pixel, method='mode', mode_option='sex')
    print('{0:7d}: {1:.5f} {2:.5f} {3:4d} {4:3d}'.format(i, msky, sky_std, nsky, nrej))
    plt.errorbar([i], msky, yerr=sky_std, capsize=3, color='b')

plt.xlabel('Star ID')
plt.ylabel('msky +- sky_std')
plt.grid(ls=':')
plt.show()

