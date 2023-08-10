from nustar_pysolar import planning, io
from imp import reload
reload(planning)
import astropy.units as u

fname = io.download_occultation_times(outdir='../data/')
print(fname)

tlefile = io.download_tle(outdir='../data')
print(tlefile)
times, line1, line2 = io.read_tle_file(tlefile)

tstart = '2017-07-10T12:00:00'
tend = '2017-07-10T20:00:00'
orbits = planning.sunlight_periods(fname, tstart, tend)

pa = planning.get_nustar_roll(tstart, 0)
print("NuSTAR Roll angle for Det0 in NE quadrant: {}".format(pa))

offset = [-190., -47.]*u.arcsec

from astropy.coordinates import SkyCoord
for ind, orbit in enumerate(orbits):
    midTime = (0.5*(orbit[1] - orbit[0]) + orbit[0])
    sky_pos = planning.get_sky_position(midTime, offset)
    print("Orbit: {}".format(ind))
    print("Orbit start: {} Orbit end: {}".format(orbit[0].isoformat(), orbit[1].isoformat()))
    print('Aim time: {} RA (deg): {} Dec (deg): {}'.format(midTime.isoformat(), sky_pos[0], sky_pos[1]))
    skyfield_pos = planning.get_skyfield_position(midTime, offset, load_path='../data')
    print('SkyField Aim time: {} RA (deg): {} Dec (deg): {}'.format(midTime.isoformat(), skyfield_pos[0], skyfield_pos[1]))
    skyfield_ephem = SkyCoord(skyfield_pos[0], skyfield_pos[1])
    sunpy_ephem = SkyCoord(sky_pos[0], sky_pos[1])
    print("")
    print("Offset between SkyField and Astropy: {} arcsec".format(skyfield_ephem.separation(sunpy_ephem).arcsec))
    print("")
         

from astropy.coordinates import SkyCoord
from datetime import timedelta


for ind, orbit in enumerate(orbits):
    midTime = orbit[0]
    while(midTime < orbit[1]):
        
        sky_pos = planning.get_sky_position(midTime, offset)

        skyfield_pos = planning.get_skyfield_position(midTime, offset, load_path='../data', parallax_correction=True)

        skyfield_ephem = SkyCoord(skyfield_pos[0], skyfield_pos[1])
        sunpy_ephem = SkyCoord(sky_pos[0], sky_pos[1])
        print('Offset between parallax-corrected positions and Astropy/Sunpy is {} arcsec'.format(
            skyfield_ephem.separation(sunpy_ephem).arcsec)
             )
        midTime += timedelta(seconds=100)
        
    break



