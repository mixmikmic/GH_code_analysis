get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from astropy.io import fits
from scipy.optimize import curve_fit

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})

# read the SNe data in
file = 'data/J_ApJ_716_712_tableb2.dat'


df = pd.read_table(file, delimiter='|', skiprows=7, header=None,
                   names=['SNe', 'S2N', 'Z', 'Bmag', 'Bmag_e', 'x1', 'x1_e', 'c', 'c_e', 'mu', 'mu_e', 'ref', 'fail'])

df



ok = np.isfinite(df['mu']) # you'll want to make this cut to get rid of missing data!

DEG = 1 # this is the order of polynomial you want to fit
fit = np.polyfit(df['XAXIS'][ok], df['YAXIS'][ok], DEG)

plt.scatter(df['XAXIS'][ok], df['YAXIS'][ok]) # plot the data again (as above)

# now plot the FIT to the data...
plt.plot(df['XAXIS'][ok], np.polyval(fit, df['XAXIS'][ok]), 
         color='red', lw=3)


fit2 = np.polyfit(np.log10(df['XAXIS'][ok]), df['YAXIS'][ok], 1)

plt.scatter(df['XAXIS'][ok], df['YAXIS'][ok]) # plot the data again (as above)

# now plot the FIT to the data...
plt.plot(df['XAXIS'][ok], np.polyval(fit2, np.log10(df['XAXIS'][ok])), 
         color='red', lw=3)






# STEP 1: make a method that produces a function

def gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

# read the data in, just like last week...
dfile = 'data/spec-3819-55540-0186.fits'

hdulist = fits.open(dfile)
tbl = hdulist[1].data
hdr = hdulist[0].header
# tbl.columns
flux = tbl['flux']

# SDSS spectra have many parameters in their "header" that define the properties of the spectrum.
# We'll use 2 of these to figure out the wavelength!
hdr

# here is how you create the "log-linear" wavelength data using these header keywords
wave = 10. ** (np.arange(0,len(flux)) * hdr['COEFF1'] + hdr['COEFF0'])

# this may be useful to you someday!! Remember it

plt.figure(figsize=(11,5))
plt.plot( )
plt.xlim(5000,6000)

p0 = (1, 2, 3, 4) # LOOK at the data above for this gaussian peak, put in good guesses!

# pick some limits within a few times the width of the peak, so to avoid (trying to) 
# fit the WHOLE spectrum with a single gaussian
WMIN = 4000
WMAX = 9000


x = np.where((wave > WMIN) & (wave < WMAX))
fit, cov = curve_fit(gaus, wave[x], flux[x], p0=p0)

plt.figure(figsize=(11,5))
plt.plot(wave, flux)
plt.plot(wave, gaus(wave, *fit)) # this *fit is a trick to explode all the parameters of "fit" in to "gauss"



