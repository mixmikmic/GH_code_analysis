get_ipython().run_cell_magic('html', '', '<center><img src="https://omundy.files.wordpress.com/2012/04/i-will-not-write-any-more-bad-code.gif" \nalt="Jokes"></center>')

import os

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')

from astropy.io.ascii import read
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import fitsio
import seaborn

# Setting convenient paths.
pypath = os.path.join(os.sep, 'global', 'work', 'projects', 'legacysurvey', 'legacycentrals')
meertcat = os.path.join(pypath, 'meert_et_al_data_tables_v2')
mendelcat = os.path.join(pypath, 'UPenn_PhotDec_Mstar_mlMendel14.dat')
wisccat = os.path.join(pypath, 'UPenn_PhotDec_Mstar_mlPCAWiscM11.dat')

# Read the Mendel catalog
columns = ('GalCount', 'FlagSerExp', 'Mstar_Tab5_Pymorph',
           'Mstar_Tab5_Truncated', 'Mstar_Tab3_Pymorph',
           'Mstar_Tab3_Truncated', 'Mstar_Tab5_Mendel',
           'Mstar_Tab3_Mendel', 'Mstar_Tab5_cModel',
           'Mstar_Tab3_cModel')
dtype = np.dtype([(col, np.float) for col in columns])
allmendel = np.loadtxt(mendelcat, dtype=dtype)

# Trim the Mendel catalog. Here we simply remove the bad flags.
keep = (np.where(allmendel['FlagSerExp'] == 0))[0]
print('Keeping {} / {} measurements in the Mendel catalog.'.format(len(keep), len(allmendel)))
mendel = allmendel[keep]

# Read the parent Meert catalog to get ra, dec and other info.
upennpath = os.path.join(pypath, 'meert_et_al_data_tables_v2')
upenncatfile = os.path.join(upennpath, 'UPenn_PhotDec_CAST.fits')
upenncat = fitsio.read(upenncatfile, ext=1, rows=keep, upper=True)

# Reading in all the RedMaPPer iSEDfit catalog.
rmpath = os.path.join(os.sep, 'global', 'work', 'projects', 'redmapper')
rmcatfile = os.path.join(rmpath, 'redmapper_isedfit_v5.10_centrals.fits.gz')
rminfo = fitsio.FITS(rmcatfile)
rmcat = rminfo[1].read(columns=['MEM_MATCH_ID', 'Z', 'RA', 'DEC', 'MSTAR_50', 'MSTAR_AVG', 'MSTAR_ERR'])

matched = os.path.join(os.sep, 'home', 'kjnapes')
mcentrals = os.path.join(matched, 'tractor_redmapper_isedfit_matches_centrals.fits')
matchedinfo = fitsio.FITS(mcentrals)
matchedcat = matchedinfo[1].read(columns=['RA', 'DEC'])

#rcoord = SkyCoord(ra=matchedcat['RA']*u.degree, dec=matchedcat['DEC']*u.degree)
#rmcoord = SkyCoord(ra=rmcat['RA']*u.degree, dec=rmcat['DEC']*u.degree)
#idx, sep2d, dist3d = rcoord.match_to_catalog_sky(rmcoord, nthneighbor=1)

#gd = np.where(sep2d < 0.001 * u.arcsec)[0]
#print(len(gd))

# WARNING! This cell is huge. It takes a long time to run.
satellites = os.path.join(rmpath, 'redmapper_isedfit_v5.10_satellites.fits.gz')
satinfo = fits.open(satellites)

#rcoord = SkyCoord(ra=matchedcat['RA']*u.degree, dec=matchedcat['DEC']*u.degree)
#rmcoord = SkyCoord(ra=satinfo[1].data['RA']*u.degree, dec=satinfo[1].data['DEC']*u.degree)
#idx, sep2d, dist3d = rcoord.match_to_catalog_sky(rmcoord, nthneighbor=1)

#gd = np.where(sep2d < 0.001 * u.arcsec)[0]
#print(len(gd))

rcoord = SkyCoord(ra=satinfo[1].data['RA']*u.degree, dec=satinfo[1].data['DEC']*u.degree)
rmcoord = SkyCoord(ra=rmcat['RA']*u.degree, dec=rmcat['DEC']*u.degree)
idx, sep2d, dist3d = rcoord.match_to_catalog_sky(rmcoord, nthneighbor=1)

gd = np.where(sep2d < 0.001 * u.arcsec)[0]
#print(satinfo[1].data['RA'][gd]-rmcat['RA'][idx[gd]])

overlap=open("overlap.txt","w")
for val in zip(rmcat['MEM_MATCH_ID'][idx[gd]], rmcat['RA'][idx[gd]],rmcat['DEC'][idx[gd]]):
    overlap.write('{}, {}, {}\n'.format(str(val[0]), val[1], val[2]))
overlap.close()

# Reading in the RA, Dec, and helioZ of Kravtsov's selected objects.
kravsources = os.path.join(os.sep, 'home','kjnapes', 'siena-astrophysics', 'research', 'massivepilot',
                           'kravsources.txt')
sourceRA, sourceDEC = np.loadtxt(kravsources, unpack=True, usecols=(1,2))
name = np.genfromtxt(kravsources, dtype='U', usecols=0)

#rcoord = SkyCoord(ra=satinfo[1].data['RA']*u.degree, dec=satinfo[1].data['DEC']*u.degree)
#kravcoord = SkyCoord(ra=sourceRA*u.degree, dec=sourceDEC*u.degree)
#idx, sep2d, dist3d = kravcoord.match_to_catalog_sky(rcoord, nthneighbor=1)

# Applying a limiting tolerance to matches. 30 arcseconds is a reasonable radius.
gd = np.where(sep2d < 3 * u.arcsec)[0]
print(len(gd))
print(name[gd])
print(satinfo[1].data['RA'][idx[gd]])
print(satinfo[1].data['DEC'][idx[gd]])
print(satinfo[1].data['Z'][idx[gd]]) # The redshifts are a little off...
print(satinfo[1].data['MSTAR_50'][idx[gd]])
print(satinfo[1].data['MSTAR_AVG'][idx[gd]])
print(satinfo[1].data['MSTAR_ERR'][idx[gd]])

# Cross-matching catalogs
rcoord = SkyCoord(ra=rmcat['RA']*u.degree, dec=rmcat['DEC']*u.degree)
kravcoord = SkyCoord(ra=sourceRA*u.degree, dec=sourceDEC*u.degree)
idx, sep2d, dist3d = kravcoord.match_to_catalog_sky(rcoord, nthneighbor=1)

# Applying a limiting tolerance to matches. 30 arcseconds is a reasonable radius.
gd = np.where(sep2d < 30 * u.arcsec)[0]

# Looking at the indices and properties of the matches. 
print(gd)
print(name[gd])
print(rmcat['RA'][idx[gd]])
print(rmcat['DEC'][idx[gd]])
print(rmcat['Z'][idx[gd]]) # The redshifts are a little off...
print(rmcat['MSTAR_50'][idx[gd]])
print(rmcat['MSTAR_AVG'][idx[gd]])
print(rmcat['MSTAR_ERR'][idx[gd]])

# Generate a custom FITS file for matching purposes
from astropy.table import Table, Column
out = Table()
out.add_column(Column(name='RA', data=rmcat['RA'][idx[gd]]))
out.add_column(Column(name='DEC', data=rmcat['DEC'][idx[gd]]))
out.write(os.path.join(os.sep, 'home','kjnapes', 'siena-astrophysics', 'research', 'massivepilot',
                           'rmmatches.fits'), overwrite=True) # Clobber is deprecated

# Cross-matching catalogs
rcoord = SkyCoord(ra=upenncat['RA']*u.degree, dec=upenncat['DEC']*u.degree)
kravcoord = SkyCoord(ra=sourceRA*u.degree, dec=sourceDEC*u.degree)
idx, sep2d, dist3d = kravcoord.match_to_catalog_sky(rcoord, nthneighbor=1)

# Applying a limiting tolerance to matches
gdpy = np.where(sep2d < 30 * u.arcsec)[0]
print(len(gdpy))

print(name[gdpy])
print(upenncat['RA'][idx[gdpy]])
print(upenncat['DEC'][idx[gdpy]])

fig = plt.figure(figsize(18,14))

ax2 = fig.add_subplot(221)
ax2.scatter(rmcat['RA'], rmcat['DEC'], color='gray', label='RedMaPPer')
ax2.scatter(sourceRA, sourceDEC, alpha=0.9, color='white', label='Kravtsov(full)')
ax2.legend(loc='upper left')

ax1 = fig.add_subplot(222)
ax1.scatter(rmcat['RA'], rmcat['DEC'], color='gray', label='RedMaPPer')
ax1.scatter(sourceRA[gd], sourceDEC[gd], alpha=0.9, color='white', label='Kravtsov (matched)')
ax1.legend(loc='upper left')

ax3 = fig.add_subplot(223)
ax3.scatter(upenncat['RA'], upenncat['DEC'], color='black', label='PyMorph')
ax3.scatter(sourceRA, sourceDEC, alpha=0.9, color='white', label='Kravtsov (full)')
ax3.legend(loc='upper left')

ax4 = fig.add_subplot(224)
ax4.scatter(upenncat['RA'], upenncat['DEC'], color='black', label='PyMorph')
ax4.scatter(sourceRA[gdpy], sourceDEC[gdpy], alpha=0.9, color='white', label='Kravtsov (matched)')
ax4.legend(loc='upper left')

from IPython.display import IFrame
IFrame('cutouts.html', width=700, height=1000)

# Read the parent Meert catalog to get ra, dec and other info.
upenncatfile = os.path.join(meertcat, 'UPenn_PhotDec_CAST.fits')
upenncat = fitsio.read(upenncatfile, ext=1, rows=keep, upper=True)

rmpath = os.path.join(os.sep, 'global', 'work', 'projects', 'redmapper')
rmcatfile = os.path.join(rmpath, 'redmapper_isedfit_v5.10_centrals.fits.gz')
rminfo = fitsio.FITS(rmcatfile)
rmcat = rminfo[1].read(columns=['Z', 'RA', 'DEC', 'LAMBDA_CHISQ', 'MSTAR_50', 'MSTAR_ERR', 'IMAG'])

# Cross-matching RedMaPPer and PyMorph
rcoord = SkyCoord(ra=rmcat['RA']*u.degree, dec=rmcat['DEC']*u.degree)
upenn = SkyCoord(ra=upenncat['RA']*u.degree, dec=upenncat['DEC']*u.degree)
idx, sep2d, dist3d = rcoord.match_to_catalog_sky(upenn, nthneighbor=1)

# Applying a limiting tolerance to matches
gd = np.where(sep2d < 3 * u.arcsec)[0]
print(len(gd))

from astropy.table import Table, Column
out = Table()
out.add_column(Column(name='LAMBDA', data=rmcat['LAMBDA_CHISQ'][idx[gd]]))
out.add_column(Column(name='DEC', data=rmcat['DEC'][idx[gd]]))
out.write(os.path.join(os.sep, 'home','kjnapes', 'siena-astrophysics', 'research', 'massivepilot',
                           'rmmatches.fits'), overwrite=True) # Clobber is deprecated

