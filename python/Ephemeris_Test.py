from nustar_pysolar import planning
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_body, solar_system_ephemeris, get_body_barycentric, SkyCoord

fname = planning.download_occultation_times(outdir='../data/')
print(fname)

tstart = '2017-05-19T01:43:00'
tend = '2017-05-19T11:43:00'
orbits = planning.sunlight_periods(fname, tstart, tend)
solar_system_ephemeris.set('jpl') 

from skyfield.api import Loader
load = Loader('../data')
ts = load.timescale()
planets = load('jup310.bsp')

dt = 0.

# Using JPL Horizons web interface at 2017-05-19T01:34:40
horizon_ephem = SkyCoord(*[193.1535, -4.01689]*u.deg)


for orbit in orbits:
    tstart = orbit[0]
    tend = orbit[1]
    print()
#    print('Orbit duration: ', tstart.isoformat(), tend.isoformat())
    on_time = (tend - tstart).total_seconds()
    
    point_time = tstart + 0.5*(tend - tstart)
    print('Time used for ephemeris: ', point_time.isoformat())
    
    astro_time = Time(point_time)
    solar_system_ephemeris.set('jpl')


    jupiter = get_body('Jupiter', astro_time)
    
    jplephem = SkyCoord(jupiter.ra.deg*u.deg, jupiter.dec.deg*u.deg)
    
    # Switch to the built in ephemris
    solar_system_ephemeris.set('builtin')
    jupiter = get_body('Jupiter', astro_time)
    
    builtin_ephem = SkyCoord(jupiter.ra.deg*u.deg, jupiter.dec.deg*u.deg)
    
    t = ts.from_astropy(astro_time)
    jupiter, earth = planets['jupiter'], planets['earth']
    astrometric = earth.at(t).observe(jupiter)
    ra, dec, distance = astrometric.radec()
    radeg = ra.to(u.deg)
    decdeg = dec.to(u.deg)
    skyfield_ephem = SkyCoord(radeg, decdeg)
    
    
    print()
    print('Horizons offset to jplephem: ', horizon_ephem.separation(jplephem))
    print()
    print('Horizons offset to "built in" ephemeris: ', horizon_ephem.separation(builtin_ephem))
    print()
    print('Horizons offset to Skyfield ephemeris: ', horizon_ephem.separation(skyfield_ephem))

    print()
    break

dt = 0.

for orbit in orbits:
    tstart = orbit[0]
    tend = orbit[1]
    print()
    on_time = (tend - tstart).total_seconds()
    
    point_time = tstart + 0.5*(tend - tstart)
    print('Time used for ephemeris: ', point_time.isoformat())
    
    astro_time = Time(point_time)
    solar_system_ephemeris.set('jpl')


    jupiter = get_body('Jupiter', astro_time)
    
    jplephem = SkyCoord(jupiter.ra.deg*u.deg, jupiter.dec.deg*u.deg)
    
    # Switch to the built in ephemris
    solar_system_ephemeris.set('builtin')
    jupiter = get_body('Jupiter', astro_time)
    
    builtin_ephem = SkyCoord(jupiter.ra.deg*u.deg, jupiter.dec.deg*u.deg)
    
    t = ts.from_astropy(astro_time)
    jupiter, earth = planets['jupiter'], planets['earth']
    astrometric = earth.at(t).observe(jupiter)
    ra, dec, distance = astrometric.radec()
    radeg = ra.to(u.deg)
    decdeg = dec.to(u.deg)
    skyfield_ephem = SkyCoord(radeg, decdeg)
    
    
    print()
    print('Skyfield offset to jplephem: ', skyfield_ephem.separation(jplephem))
    print()
    print('Skyfield offset to "built in" ephemeris: ', skyfield_ephem.separation(builtin_ephem))

    print()


