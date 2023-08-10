from nustar_pysolar import planning, io
import astropy.units as u

fname = io.download_occultation_times(outdir='../data/')
print(fname)

tlefile = io.download_tle(outdir='../data')
print(tlefile)
times, line1, line2 = io.read_tle_file(tlefile)

tstart = '2017-06-10T12:00:00'
tend = '2017-06-10T20:00:00'
orbits = planning.sunlight_periods(fname, tstart, tend)

pa = planning.get_nustar_roll(tstart, 0)
print("NuSTAR Roll angle for Det0 in NE quadrant: {}".format(pa))

offset = [-190., -47.]*u.arcsec

for ind, orbit in enumerate(orbits):
    midTime = (0.5*(orbit[1] - orbit[0]) + orbit[0])
    sky_pos = planning.get_sky_position(midTime, offset)
    print("Orbit: {}".format(ind))
    print("Orbit start: {} Orbit end: {}".format(orbit[0].isoformat(), orbit[1].isoformat()))
    print('Aim time: {} RA (deg): {} Dec (deg): {}'.format(midTime.isoformat(), sky_pos[0], sky_pos[1]))
    print("")

aim_time = '2016-07-26T19:53:15.00'
offset = [1000, 150]*u.arcsec
sky_pos = planning.get_sky_position(aim_time, offset)
print(sky_pos)

np = planning.get_nustar_roll(aim_time, 0)
print(np)

