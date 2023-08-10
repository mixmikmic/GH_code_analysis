from astropy.time import Time
from astropy.io import fits
from glob import glob

import exoplanet

star = exoplanet.Star('TrES-3')

star.planet.transittime

star.planet.transit_duration

print("Depth: {:.02%}".format(star.planet.transit_depth))

start_time = Time('2017-06-01')
end_time = Time('2017-06-10')

for t in star.planet.transits_in_range(start_time, end_time):
    print(t.ingress)

fits_list = glob('/var/panoptes/images/fields/20170607T071053/*.fz', recursive=True)

fits_list.sort()
len(fits_list)

for f in fits_list:
    if f.endswith('.fz'):
        ext = 1
    else:
        ext = 0
        
    t1 = fits.getval(f, 'DATE-OBS', ext=ext)
    if star.planet.in_transit(t1):
        print(t1, f)
    else:
        print('.', end='')

