import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

get_ipython().run_line_magic('matplotlib', 'inline')

datadir = 'Sampledata/photometry/'
sciencedir = datadir+'science/'

# Create a filepath "glob" (list) of all bias frames
bias_subdir = 'calibration/bias/'
bias_files = glob.glob(datadir+bias_subdir+'*.fits')

bias_data = [fits.getdata(b) for b in bias_files]
master_bias = np.mean(bias_data, axis=0)  # axis=0 retains the shape of the original array

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
hist = ax1.hist(master_bias.flatten(), bins=1000, range=[2000,2500])
img = plt.imshow(master_bias, cmap='gray', vmin=2000, vmax=2500, origin='lower')#,ax=ax2)
plt.colorbar()

master_bias_head = fits.getheader(bias_files[0])
fits.writeto(sciencedir+'master_bias.fits', master_bias, header=master_bias_head, clobber=True)

took_darks = False

if took_darks:
    dark_subdir = 'calibration/dark/'
    dark_files = glob.glob(datadir+dark_subdir+'*.fits')

    # We must remove the bias to determine the contribution from only the dark counts
    dark_data = [fits.getdata(d)-master_bias for d in dark_files]
    master_dark = np.average(dark_data, axis=0)
    
    master_dark_head = fits.getheader(dark_files[0])
    fits.writeto(sciencedir+dark_subdir+'master_dark.fits', master_dark, 
                 header=master_dark_head, clobber=True)
else:
    # Just create an array of zeros with the same dimensions as our data.
    # We'll use the master bias to determine these dimensions
    data_shape = np.shape(master_bias)
    master_dark = np.zeros(shape=data_shape)
    
    fits.writeto(sciencedir+'master_dark.fits', master_dark, clobber=True)

flat_subdir = 'calibration/flat/'
flat_files = glob.glob(datadir+flat_subdir+'*.fits')

# Again, we need to remove the bias counts from all flat field exposures to 
# determine the contribution from just the flats
# flat_data = [fits.getdata(f)-master_bias for f in flat_files]
flat_data = [np.subtract(fits.getdata(f),master_bias) for f in flat_files]
master_flat = np.median(flat_data, axis=0)
master_flat = master_flat/np.median(master_flat)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
hist = ax1.hist(master_flat.flatten(), bins=1000, range=[0.5,1.5])
img = plt.imshow(master_flat, cmap='gray',vmin=0.7, vmax=1.1, origin='lower')
plt.colorbar()

master_flat_head = fits.getheader(flat_files[0])
fits.writeto(sciencedir+'master_flat.fits', master_flat, 
             header=master_flat_head, clobber=True)

science_files = glob.glob(sciencedir+'n*.fits')
science_data = [fits.getdata(s) for s in science_files]

calibrated_data = (science_data - master_bias - master_dark)/master_flat

calibrated_data_head = fits.getheader(science_files[0])
fits.writeto(sciencedir+'calibrated.fits', calibrated_data, 
             header=calibrated_data_head, clobber=True)

# vmin and vmax can be determined by plotting a histogram of the flattened image data
plt.imshow(np.mean(calibrated_data, axis=0), cmap='gray', norm=LogNorm(), vmin=600, vmax=1500, origin='lower')
plt.colorbar()

