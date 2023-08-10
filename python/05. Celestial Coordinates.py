from astropy import units as u
from astropy.coordinates import SkyCoord

c = SkyCoord(ra=10.625*u.degree, dec=41.2*u.degree)
c = SkyCoord(10.625, 41.2, unit='deg')
c = SkyCoord('00h42m30s', '+41d12m00s')
c = SkyCoord('00h42.5m', '+41d12m')
c = SkyCoord('00 42 30 +41 12 00', unit=(u.hourangle, u.deg))
c = SkyCoord('00:42.5 +41:12', unit=(u.hourangle, u.deg))
c

c.ra 

c.ra.hour

c.dec  

c.dec.radian

c.to_string('hmsdms')

c.galactic  

c.transform_to('fk5')

from astropy.coordinates import FK5
c.transform_to(FK5(equinox='J1975'))

from astropy.coordinates import get_icrs_coordinates

cm42 = get_icrs_coordinates('m42')
cm42

from astropy.coordinates import get_constellation

get_constellation(cm42)

from astropy.coordinates import get_sun
from astropy.time import Time

get_sun(Time('1999-01-01T00:00:00.123456789'))

c = SkyCoord(x=1, y=2, z=3, unit='kpc', representation='cartesian')
c

c.representation = 'cylindrical'
c

c1 = SkyCoord(ra=10*u.degree, dec=9*u.degree, distance=10*u.pc)
c2 = SkyCoord(ra=11*u.degree, dec=10*u.degree, distance=11.5*u.pc)

c1.separation_3d(c2)

c.cartesian.x

c = SkyCoord([33,24], [41,23], unit='deg')
c

