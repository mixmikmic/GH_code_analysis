get_ipython().magic('matplotlib inline')
import numpy as np
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic
from lsst.sims.catalogs.db import CatalogDBObject
from lsst.sims.catUtils.baseCatalogModels import *
from lsst.sims.catUtils.exampleCatalogDefinitions import *
from lsst.sims.utils import ObservationMetaData

# Parameters we need to set

# Magnitudes of gray cloud extinction one should be able to see through
max_extinction = 1.0

# Precision of the measured extinction
extinction_precision = 0.2

# Dynamic range of all-sky camera. Min=observed saturation mag. 
# Max=faint end of catalog (or where Gaussian error no longer true)
mag_range = [7, 12]

# Angular resolution requirement, diameter
ang_resolution = 1. #  Degrees

# Time resolution
time_resolution = 30. #  Seconds

# Seeing requirement

# Sky brightness requirement

# Airmass requrement
airmass_limit = 2.0

# Atmospheric extinction
kAtm = 0.15

# filter to use
filtername = 'g'

# 5-sigma limiting depth of a point source at zenith on a clear night for the camera.
m5 = 10.

# Coordinates at the galactic pole
c = SkyCoord(Galactic, l=0.*u.degree , b=-90*u.degree)

c.icrs.ra.deg, c.icrs.dec

# Radius to query, in degrees
boundLength = ang_resolution/2.

colnames = ['raJ2000', 'decJ2000', 'umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']
constraint = 'gmag < %f' % (np.max(mag_range)+ max_extinction + airmass_limit*kAtm)

# dbobj = CatalogDBObject.from_objid('allstars')
dbobj = CatalogDBObject.from_objid('brightstars')

obs_metadata = ObservationMetaData(boundType='circle',
                                   pointingRA=c.icrs.ra.deg,
                                   pointingDec=c.icrs.dec.deg,
                                   boundLength=boundLength, mjd=5700)

t = dbobj.getCatalog('ref_catalog_star', obs_metadata=obs_metadata)

stars = t.db_obj.query_columns(colnames=colnames, obs_metadata=obs_metadata,
                               constraint=constraint, limit=1e6, chunk_size=None)
stars = [chunk for chunk in stars][0]

print 'found %i stars within %f degrees of the galactic pole' % (stars.size, ang_resolution/2.)

observed_mags = stars[filtername+'mag'] + max_extinction + airmass_limit*kAtm
good = np.where( (observed_mags > np.min(mag_range)) & (observed_mags < np.max(mag_range)))
observed_mags = observed_mags[good]
print 'Able to measure %i stars though %.1f mags of cloud extinction' % (observed_mags.size, max_extinction)

snr =  5.*10.**(-0.4*(observed_mags-m5))
mag_uncertanties = 1./snr
# If each star is an independent measure of the zeropoint, the final 
# SNR on the zeropoint will be the combination of all the individual star SNRs.
final_snr = np.sqrt(np.sum(snr**2))

zp_uncertainty = 1./final_snr
print 'final zeropoint uncertainty = %f mags' % zp_uncertainty







