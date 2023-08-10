import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, Column
get_ipython().magic('matplotlib inline')



# Read the catalogs:
#   /global/work/data/sdss/dr12/DR12Q.fits
#   /global/work/decam/release/dr3/external/survey-dr3-DR12Q.fits
sdss = fits.getdata("/Users/ioannis/tmp/DR12Q.fits", 1)

dcal = fits.getdata("/Users/ioannis/tmp/survey-dr3-DR12Q.fits", 1)
#dcal.columns
sdss.columns

index = np.where(dcal['OBJID']>0)[0]
print len(index), len(dcal)
qso = dcal[index]
sqso = sdss[index]

plt.scatter(qso['RA'], qso['DEC'])

# Make some basic plots
#   * ra, dec - full SDSS and matched SDSS/DR3
#   * redshift histogram
# Select a subset of QSOs and write them out

plt.hist(sqso['Z_PIPE'], bins=50)
plt.xlim(1,6)

# Upload the output catalog; inspect the images and spectra.
ww = np.where(sqso['Z_PIPE']>4)[0]
out = Table()
out.add_column(Column(name='RA', data=qso['RA'][ww]))
out.add_column(Column(name='DEC', data=qso['DEC'][ww]))
out.write('/Users/ioannis/tmp/hiz-qsos.fits')

# Read the catalogs:
#   /global/work/projects/redmapper/redmapper_v5.10_dr3.fits
#   /global/work/projects/redmapper/redmapper_isedfit_v5.10_centrals.fits.gz

# Visually inspect 2-3 clusters in the viewer

# Make a plot of cluster richness vs central galaxy stellar mass

