get_ipython().magic('matplotlib inline')
from gammapy.data import EventList
from gammapy.image import SkyImage
from astropy.io import fits

fits_name = 'sim_events_000002.fits'
hdulist = fits.open(fits_name)
prihdr = hdulist[1].header

ra_pnt = prihdr['RA_PNT']
dec_pnt = prihdr['DEC_PNT']
events = EventList.read(fits_name)

deg = 8
binsz = 0.2
npix = int(deg/binsz)
image = SkyImage.empty(nxpix = npix, nypix=npix, binsz=binsz, xref = ra_pnt, yref=dec_pnt, coordsys='CEL', proj='AIT')

image.fill_events(events)

image.show()



