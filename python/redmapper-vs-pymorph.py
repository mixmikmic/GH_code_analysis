import os
import warnings

from astropy.io import fits
import numpy as np
import mpl_toolkits
import matplotlib.pyplot as plt

import seaborn as sns
import fitsio
from desitarget.targetmask import desi_mask, bgs_mask
from desiutil.plots import init_sky, plot_healpix_map, plot_grid_map, plot_sky_circles, plot_sky_binned, prepare_data
import emcee

get_ipython().magic('matplotlib inline')

# Trims data; outputs trimmed data into a fits file
CUTS = True
QAPLOTS = True

path = '/Users/kevinnapier/research/dmhalos/'
file = 'tractor-redmapper_isedfit_v5.10_centrals.fits' # dr3data; which one should I use?
centrals = 'redmapper_isedfit_v5.10_centrals.fits' 

data = fits.getdata(os.path.join(path, file))
centralsdata = fits.getdata(os.path.join(path, centrals))

if CUTS == True:
    cuts = np.where(data['OBJID'] == -1)
    newdata = np.delete(centralsdata, cuts[0])
    fits.writeto(os.path.join(path,'tractor_matched_centrals.fits'), newdata, overwrite=True)

guts = os.path.join(path, 'tractor_matched_centrals.fits')
matched = fitsio.read(os.path.join(path, 'tractor_matched_centrals.fits'))
#ii=fitsio.FITS(os.path.join(path, guts))
#print(ii[1])

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    plt.figure(figsize=(8,4))
    basemap = init_sky(galactic_plane_color='k');
    plot_sky_binned(centralsdata['RA'], centralsdata['DEC'], verbose=False, clip_lo='!1', plot_type='healpix', 
                    cmap='jet', label=r'All Centrals (sources/deg$^2$)', basemap=basemap);

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    plt.figure(figsize=(8,4))
    basemap = init_sky(galactic_plane_color='k');
    plot_sky_binned(matched['RA'], matched['DEC'], verbose=False, clip_lo='!1', plot_type='healpix', cmap='jet', 
                label=r'Matched Centrals (sources/deg$^2$)', basemap=basemap);

REDSHIFT = matched['Z']
RICHNESS = matched['LAMBDA_CHISQ']
IMAG = matched['IMAG']

sns.kdeplot(np.array(REDSHIFT), np.array(RICHNESS),cmap="Blues", shade=True, shade_lowest=False)
plt.ylim([0, 50])
plt.show()

plt.hist(REDSHIFT, bins=500)
plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io.ascii import read
import fitsio

get_ipython().magic('pylab inline')

pypath = os.path.join(os.sep, 'global', 'work', 'projects', 'legacysurvey', 'legacycentrals')
meertcat = os.path.join(pypath, 'meert_et_al_data_tables_v2')
mendelcat = os.path.join(pypath, 'UPenn_PhotDec_Mstar_mlMendel14.dat')
wisccat = os.path.join(pypath, 'UPenn_PhotDec_Mstar_mlPCAWiscM11.dat')

rmpath = os.path.join(os.sep, 'global', 'work', 'projects', 'redmapper')
rmcatfile = os.path.join(rmpath, 'redmapper_isedfit_v5.10_centrals.fits.gz')

# Read the Mendel catalog
columns = ('GalCount', 'FlagSerExp', 'Mstar_Tab5_Pymorph',
           'Mstar_Tab5_Truncated', 'Mstar_Tab3_Pymorph',
           'Mstar_Tab3_Truncated', 'Mstar_Tab5_Mendel',
           'Mstar_Tab3_Mendel', 'Mstar_Tab5_cModel',
           'Mstar_Tab3_cModel')
dtype = np.dtype([(col, np.float) for col in columns])
allmendel = np.loadtxt(mendelcat, dtype=dtype)
allmendel.dtype.names

# Keep good measurements.
keep = np.where(allmendel['FlagSerExp'] == 0)[0]
print('Keeping {} / {} measurements in the Mendel catalog.'.format(len(keep), len(allmendel)))
mendel = allmendel[keep]

# Read the parent Meert catalog to get ra, dec and other info.
upennpath = os.path.join(pypath, 'meert_et_al_data_tables_v2')
upenncatfile = os.path.join(upennpath, 'UPenn_PhotDec_CAST.fits')
upenncat = fitsio.read(upenncatfile, ext=1, rows=keep)
upenncat.dtype.names
len(upenncat)

len(mendel)

np.all((upenncat['galcount'], mendel['GalCount']))

rminfo = fitsio.FITS(rmcatfile)
rmcat = rminfo[1].read(columns=['Z', 'RA', 'DEC', 'LAMBDA_CHISQ', 'MSTAR_50', 'MSTAR_ERR'])

# Match to remapper, make plots.

fig, ax = plt.subplots()
ax.scatter(rmcat['MSTAR_50'], mendel['Mstar_Tab5_Pymorph'])

