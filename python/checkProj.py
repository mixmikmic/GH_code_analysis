get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
import healpy as hp

def raDec2Hpid(nside, ra, dec):
    """
    Assign ra,dec points to the correct healpixel.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    ra : np.array
        RA values to assign to healpixels.
    dec : np.array
        Dec values to assign to healpixels.

    Returns
    -------
    hpids : np.array
        Healpixel IDs for the input positions.
    """
    lat = np.pi/2. - dec
    hpids = hp.ang2pix(nside, lat, ra)
    return hpids

def _hpid2RaDec(nside, hpids):
    """
    Correct for healpy being silly and running dec from 0-180.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    hpids : np.array
        Array (or single value) of healpixel IDs.

    Returns
    -------
    raRet : float (or np.array)
        RA positions of the input healpixel IDs. In radians.
    decRet : float (or np.array)
        Dec positions of the input healpixel IDs. In radians.
    """

    lat, lon = hp.pix2ang(nside, hpids)
    decRet = np.pi/2. - lat
    raRet = lon

    return raRet, decRet

def healbin(ra, dec, values, nside=128, reduceFunc=np.mean, dtype=float):
    """
    Take arrays of ra's, dec's, and value and bin into healpixels. Like numpy.hexbin but for
    bins on a sphere.

    Parameters
    ----------
    ra : np.array
        RA positions of the data points.
    dec : np.array
        Dec positions of the data points.
    values : np.array
        The values at each ra,dec position.
    nside : int
        Healpixel nside resolution. Must be a value of 2^N.
    reduceFunc : function (numpy.mean)
        A function that will return a single value given a subset of `values`.

    Returns
    -------
    mapVals : np.array
        A numpy array that is a valid Healpixel map.
    """

    hpids = raDec2Hpid(nside, ra, dec)

    order = np.argsort(hpids)
    hpids = hpids[order]
    values = values[order]
    pixids = np.unique(hpids)
    pixids = np.arange(hp.nside2npix(nside))

    left = np.searchsorted(hpids, pixids)
    right = np.searchsorted(hpids, pixids, side='right')

    mapVals = np.zeros(hp.nside2npix(nside), dtype=dtype)+hp.UNSEEN

    # Wow, I thought histogram would be faster than the loop, but this has been faster!
    for i, idx in enumerate(pixids):
        mapVals[idx] = reduceFunc(values[left[idx]:right[idx]] )

    # Change any NaNs to healpy mask value
    mapVals[np.isnan(mapVals)] = hp.UNSEEN

    return mapVals

def stupidFast_RaDec2AltAz(ra, dec, lat, lon, mjd, lmst=None):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore abberation, precesion, nutation, nutrition, etc.

    Parameters
    ----------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    if lmst is None:
        lmst, last = callablecLmstLast(mjd, lon)
        lmst = lmst/12.*np.pi  # convert to rad
    ha = lmst-ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    alt = np.arcsin(sindec*sinlat+np.cos(dec)*coslat*np.cos(ha))
    az = np.arccos((sindec-np.sin(alt)*sinlat)/(np.cos(alt)*coslat))
    signflip = np.where(np.sin(ha) > 0)
    az[signflip] = 2.*np.pi-az[signflip]
    return alt, az

hdulist = fits.open('ut012716.0100.long.M.fits')

image = hdulist[0].data

image

# Coordinates are hour angle and declination.
coords = np.genfromtxt('ut012716.0100.long.M.xxyy', dtype=[int,int,float,float])

plt.imshow(image, vmin=image.min(), vmax=image.min()+100)
#plt.figure()
#ax = plt.subplot(111, projection = 'mollweide')
#ax.plot(coords[:,2],coords[:,3])

print image.shape, coords.shape

biasLevel = np.median(image[0:500,0:500])
biasLevel

healMap = healbin(np.radians(coords['f2']), np.radians(coords['f3']), 
                  image[coords['f1'],coords['f0'] ], reduceFunc=np.median )

hp.mollview(healMap, max=2134)

healMap = healbin(np.radians(coords['f2']), np.radians(coords['f3']), 
                  image[coords['f1'],coords['f0'] ],nside=32, reduceFunc=np.median  )
hp.mollview(healMap, max=2134)



ack = healMap-biasLevel
hp.mollview(ack)
mask = np.where(ack < 10)
ack = np.log10(ack)
ack[mask] = hp.UNSEEN
hp.mollview(ack, max=2)

#from lsst.sims.utils import Site
#site = Site('LSST')

mjd = hdulist[0].header['MJD-OBS']
import ephem
obs = ephem.Observer()
obs.lon = -1.23480997381 #site.longitude_rad
obs.lat = -0.52786436029 #site.latitude_rad
obs.elevation = 2650.0 #site.height
doff = ephem.Date(0)-ephem.Date('1858/11/17')
obs.date = mjd - doff
lst = obs.sidereal_time()
#print site.longitude_rad, site.latitude_rad, site.height

ra = np.radians(coords['f2']) - lst
while ra.min() < 0:
    ra += 2.*np.pi
ra = ra % (2.*np.pi)

nside = 32
healMap = healbin(ra, np.radians(coords['f3']), 
                  image[coords['f1'],coords['f0'] ],nside=nside, reduceFunc=np.median  )
healMap -= biasLevel
healMap = -2.5*np.log10(healMap)
cutoff = -2.5*np.log10(10.)  # Require 10 counts above bias level to consider pixel observed. Why 10, why not!
healMap[np.isnan(healMap)] = hp.UNSEEN
healMap[np.where(healMap > cutoff)] = hp.UNSEEN
hp.mollview(healMap)#, max=2134)

hpra, hpdec = _hpid2RaDec(nside, np.arange(np.size(healMap)))
alt, az = stupidFast_RaDec2AltAz(hpra, hpdec, obs.lat, obs.lon, mjd, lmst=lst)

airmass = 1./np.cos(np.pi/2-alt)

# let's generate some output
# Normally would also want G and B filters as well.
print '#healpix id (nside=%i), median R, airmass, mjd ' % nside
good = np.where((healMap != hp.UNSEEN) & (alt > 0))[0]
for i in good[0:10]:
    print '%i, %f, %f, %f' % (i, healMap[i], airmass[i], mjd)





