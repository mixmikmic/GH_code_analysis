get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import glob
import os
import sys
import time

from read_stars import read_manual_stars

from lsst.all_sky_phot.wcs import wcs_zea, wcs_refine_zea, Fisheye, distortion_mapper, distortion_mapper_looper
from lsst.all_sky_phot import phot_night, readcr2, readYBC, radec2altaz

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, Longitude, Latitude
import astropy.units as u
from astropy.time import Time
from astropy.table import Table, vstack
from astropy import units as u

from photutils import CircularAperture

# Read in stars measured off several exposures
stars = read_manual_stars('starcoords.dat')
stars

# Fit a Zenith Equal Area projection.
fun = wcs_zea(stars['x'], stars['y'], stars['alt'], stars['az'], crpix1=2.87356521e+03, crpix2=1.98559533e+03)
x0 = np.array([2.87356521e+03,   1.98559533e+03,  1., 1., .036,
        0.0027,  0.00295,   -0.0359])
fit_result = minimize(fun, x0)

# Convert the fit to a full WCS object
wcs_initial = fun.return_wcs(fit_result.x)

# Check the residuals
fit_x, fit_y = wcs_initial.all_world2pix(stars['az'], stars['alt'], 0)
resid_d = ((fit_x-stars['x'])**2+(fit_y-stars['y'])**2)**0.5
plt.scatter(fit_x, fit_y, c=resid_d)
cb = plt.colorbar()
cb.set_label('Residual (pixels)')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Rough Initial WCS Residuals')

# Let's see what the result looks like
x, y = np.meshgrid(np.arange(1000,4000, 1), np.arange(500, 3500, 1))
x = x.ravel()
y = y.ravel()
az, alt = wcs_initial.all_pix2world(x,y, 0)
plt.hexbin(x,y, alt)
plt.xlabel('x position')
plt.ylabel('y position')
cb = plt.colorbar()
cb.set_label('Altitude (deg)')
plt.figure()
plt.hexbin(x,y, az)
plt.xlabel('x position')
plt.ylabel('y position')
cb = plt.colorbar()
cb.set_label('Azimuth (deg)')

# Now to run photometry on a full image and refine the WCS solution
# Load the Yale bright star catalog
ybc = readYBC()

filename = 'ut012716/ut012716.0130.long.cr2'
#filename = 'ut012516/ut012516.0322.long.cr2'
im, header = readcr2(filename)
# Combine the RGB into a single image
sum_image = np.sum(im, axis=2).astype(float)
# Run detection and photometry
phot_tables = phot_night([filename], savefile=None, progress_bar=False)
phot_appertures = CircularAperture( (phot_tables[0]['xcenter'], phot_tables[0]['ycenter']), r=5.)
# guess the camera zeropoint
zp = -18.
measured_mags = -2.5*np.log10(phot_tables[0]['residual_aperture_sum'].data) - zp

# OK, let's see where we expect the stars to be
lsst_location = EarthLocation(lat=-30.2444*u.degree, lon=-70.7494*u.degree, height=2650.0*u.meter)
alt_cat, az_cat = radec2altaz(ybc['RA'], ybc['Dec'], header['mjd'], location=lsst_location)
above = np.where(alt_cat > 5.)
x_expected, y_expected = wcs_initial.all_world2pix(az_cat[above], alt_cat[above], 0.)
apertures = CircularAperture( (x_expected, y_expected), r=5.)

# Let's take a look at where the stars are, and where predicted stars are:
plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower', vmin=2, vmax=4)
plt.colorbar()

plt.figure(figsize=[20,20])
plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower', vmin=2, vmax=4)
plt.colorbar()
plt.xlim([2500,3500])
plt.ylim([2500,3500])
# Detected objects in blue
phot_appertures.plot(color='blue', lw=1.5, alpha=0.5)
# Predicted locations in green
apertures.plot(color='green', lw=3, alpha=0.75)

x0

fun = wcs_refine_zea(phot_tables[0]['xcenter'].value, phot_tables[0]['ycenter'].value, measured_mags,header['mjd'],
                     ybc['RA'].values, ybc['Dec'].values, ybc['Vmag'].values, a_order=0, b_order=0)
x0 = fun.wcs2x0(wcs_initial)
fit_result = minimize(fun, x0[0:8], method='Powell')

# Check that we did imporved the fit
fun(fit_result.x) < fun(x0[0:8])

# Let's update the expected position of reference stars and see how we did
wcs_refined = fun.return_wcs(fit_result.x)
x_expected, y_expected = wcs_refined.all_world2pix(az_cat[above], alt_cat[above], 0.)
apertures = CircularAperture( (x_expected, y_expected), r=5.)

plt.figure(figsize=[20,20])
plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower', vmin=2, vmax=4)
plt.colorbar()
plt.xlim([2500,3500])
plt.ylim([2500,3500])
phot_appertures.plot(color='blue', lw=1.5, alpha=0.5)
apertures.plot(color='green', lw=3, alpha=0.75)

# Run one night to fit, and the next to check the solution
roots = ['012616', '012716']
for root in roots:
    savefile = root+'_night_phot.npz'
    # Don't bother if already run
    if not os.path.isfile(savefile):
        files = glob.glob('ut'+root+'/*.cr2')
        phot_tables = phot_night(files, savefile=savefile)

# Load up the photometry tables from the 1st night
temp = np.load('012716_night_phot.npz')
phot_tables = temp['phot_tables'][()]
temp.close()

# Which photometry tables to use
minindx=50
maxindx=91

# Build arrays of observed and expected locations
alts = []
azs = []
mjds = []
observed_x = []
observed_y = []
observed_mjd = []
for phot_table in phot_tables[minindx:maxindx]:
    mjd = phot_table['mjd'][0]
    alt, az = radec2altaz(ybc['RA'], ybc['Dec'], mjd, location=lsst_location)
    good = np.where(alt > 0.)
    alts.append(alt[good])
    azs.append(az[good])
    mjds.append(az[good]*0+mjd)
    observed_x.append(phot_table['xcenter'].value)
    observed_y.append(phot_table['ycenter'].value)
    observed_mjd.append(phot_table['mjd'].data)

alts = np.concatenate(alts)
azs = np.concatenate(azs)
mjds = np.concatenate(mjds)
print('Predicting %i star locations' % mjds.size)
observed_x = np.concatenate(observed_x)
observed_y = np.concatenate(observed_y)
observed_mjd = np.concatenate(observed_mjd)

t0 = time.time()
filename='fit_result_%i_%i.npz' % (minindx, maxindx)
# Number of points to make in x and y
nx=70
ny=50
window=50

if not os.path.isfile(filename):
    wcs_w_shift, result = distortion_mapper_looper(observed_x, observed_y, observed_mjd, alts, azs, mjds,
                                                  wcs_refined, xmax=5796, ymax=3870, nx=nx, ny=ny,
                                                  window=window)
    # unpack variables for convienence
    yp = result['yp']
    xp = result['xp']
    xshifts = result['xshifts']
    yshifts = result['yshifts']
    distances = result['distances']
    npts = result['npts']
    np.savez(filename, xp=result['xp'], yp=result['yp'], xshifts=result['xshifts'], yshifts=result['yshifts'],
             distances=result['distances'], npts=result['npts'])
else:
    data = np.load(filename)
    yp = data['yp'].copy()
    xp = data['xp'].copy()
    xshifts = data['xshifts'].copy()
    yshifts = data['yshifts'].copy()
    distances = data['distances'].copy()
    npts = data['npts'].copy()
    data.close()
    good = np.where(~(np.isnan(xshifts)) & ~(np.isnan(yshifts)))
    wcs_w_shift = Fisheye(wcs_refined, xp[good], yp[good], xshifts[good], yshifts[good])
t1 = time.time()
print('time to map distortions = %.1f min' % ( (t1-t0)/60.))



# Look at the distortion map that we have to apply after the WCS solution
plt.scatter(xp,yp, c=xshifts, vmin=-15, vmax=10, s=40)
cb = plt.colorbar()
cb.set_label('x-shift (pix)')
plt.figure()
plt.scatter(xp,yp, c=yshifts, vmin=-15, vmax=10, s=40)
cb = plt.colorbar()
cb.set_label('y-shift (pix)')

plt.scatter(xp,yp, c=distances, s=40, vmax=3)
cb = plt.colorbar()
cb.set_label('Median distance to nearest neighbor (pix)')



x_expected, y_expected = wcs_w_shift.all_world2pix(az_cat[above], alt_cat[above], 0)
apertures = CircularAperture( (x_expected, y_expected), r=5.)

plt.figure(figsize=[20,20])
plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower', vmin=2, vmax=4)
plt.colorbar()
plt.xlim([2500,3500])
plt.ylim([2500,3500])
phot_appertures.plot(color='blue', lw=3.5, alpha=0.5)
apertures.plot(color='green', lw=3, alpha=0.5)


plt.figure(figsize=[20,20])
plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower', vmin=2, vmax=4)
plt.colorbar()
plt.xlim([2500, 4000])
plt.ylim([1500,2500])
phot_appertures.plot(color='blue', lw=1.5, alpha=0.5)
apertures.plot(color='green', lw=3, alpha=0.75)

plt.figure(figsize=[20,20])
plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower', vmin=2, vmax=4)
plt.colorbar()
plt.xlim([1000, 2500])
plt.ylim([0,1500])
phot_appertures.plot(color='blue', lw=1.5, alpha=0.5)
apertures.plot(color='green', lw=3, alpha=0.75)

# Let's check an image from a different night
#filename = 'ut012716/ut012716.0130.long.cr2'
filename = 'ut012516/ut012516.0322.long.cr2'
im, header = readcr2(filename)
# Combine the RGB into a single image
sum_image = np.sum(im, axis=2).astype(float)
# Run detection and photometry
phot_tables = phot_night([filename], savefile=None, progress_bar=False)
phot_appertures = CircularAperture( (phot_tables[0]['xcenter'], phot_tables[0]['ycenter']), r=5.)
# guess the camera zeropoint
zp = -18.
measured_mags = -2.5*np.log10(phot_tables[0]['residual_aperture_sum'].data) - zp

alt_cat, az_cat = radec2altaz(ybc['RA'], ybc['Dec'], header['mjd'], location=lsst_location)
above = np.where(alt_cat > 5.)
x_expected, y_expected = wcs_initial.all_world2pix(az_cat[above], alt_cat[above], 0.)
apertures = CircularAperture( (x_expected, y_expected), r=5.)

x_expected, y_expected = wcs_w_shift.all_world2pix(az_cat[above], alt_cat[above], 0)
apertures = CircularAperture( (x_expected, y_expected), r=5.)

plt.figure(figsize=[20,20])
plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower', vmin=3, vmax=5)
plt.colorbar()
plt.xlim([2500,3500])
plt.ylim([2500,3500])
phot_appertures.plot(color='blue', lw=3.5, alpha=0.5)
apertures.plot(color='green', lw=3, alpha=0.5)

plt.imshow(np.log10(sum_image),  cmap='Greys', origin='lower')

# OK, since this looks good, let's save the wcs solution with distortion map
wcs_w_shift.save('fisheye_wcs.npz')



