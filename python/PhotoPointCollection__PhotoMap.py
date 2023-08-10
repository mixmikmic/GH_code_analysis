## plot within the notebook
get_ipython().magic('matplotlib inline')
import warnings
## No annoying warnings
warnings.filterwarnings('ignore')
# - Astrobject Modules
from astrobject import instrument
from astropy import units, table

sdss = instrument("data/sdss_PTF10qjq_g.fits")
# let's see how it looks like
pl = sdss.show()

sdss.download_catalogue(source="sdss",column_filters={"gmag":"13..22"})

sdss.catalogue.define_around(15*units.arcsec)

mask_isolatedstars = sdss.catalogue.starmask*sdss.catalogue.isolatedmask
ra,dec = sdss.catalogue.ra[mask_isolatedstars], sdss.catalogue.dec[mask_isolatedstars]

# ra and dec are list of coordinate. 
# This means: give me a collection of photopoints for each coordinates (in radec) within an aperture of 5 arcsec.
photomap = sdss.get_photopoint(ra, dec, radius=5, runits="arcsec", wcs_coords=True)

pl = sdss.show(show=False)
pl_vor = photomap.display_voronoy(pl["ax"],"mag", cblabel="measured magnitude of isolated stars")



