from astroplan import Observer, FixedTarget
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

catalog_name = 'J/MNRAS/446/4078/'

Vizier.ROW_LIMIT = -1       # Otherwise would only show first 50 values
catalog_list = Vizier.find_catalogs(catalog_name)
catalogs = Vizier.get_catalogs(list(catalog_list.keys()))
catalog_table = catalogs[0] # This is the table with the data

RAs = catalog_table['_RAJ2000']
Decs = catalog_table['_DEJ2000']
sc = SkyCoord(ra=RAs, dec=Decs, frame='icrs')
names = catalog_table['SDSS']
types = catalog_table['Type']
gmag = catalog_table['gmag']

max_g = 17.5
bright_metal_polluted = np.array(['z' in str(t).lower() and g < max_g
                                  for t, g in zip(types.data, 
                                                  gmag.data)])

target_list = [FixedTarget(c, name="SDSS{0:s}_{1:.2f}".format(n.decode('ascii'), g)) 
               for c, n, g, b in zip(sc, names, gmag, bright_metal_polluted) if b]

print('{0} stars selected'.format(len(target_list)))

obs = Observer.at_site("APO", timezone='US/Mountain')

from astroplan import (is_observable, observability_table, 
                       AltitudeConstraint, AtNightConstraint)
from astropy.time import Time

constraints = [AltitudeConstraint(min=30*u.deg), 
               AtNightConstraint.twilight_astronomical()]

days_range = Time('2016-01-15 19:00') + np.arange(0, 365-30, 30)*u.day
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
observable_targets = []
for month, day in zip(months, days_range):
    tonight_start = obs.twilight_evening_astronomical(day, which='next')
    tonight_end = obs.twilight_morning_astronomical(day, which='next')
    table = observability_table(constraints, obs, target_list, 
                                time_range=Time([tonight_start, tonight_end]))
    seventypercent = table['fraction of time observable'] > 0.75
    seventypercent_targets = table['target name'][seventypercent].data
    print("{0}\n----------\n{1}\n".format(day.datetime.date(), 
                                          '\n'.join(sorted(seventypercent_targets, 
                                                           key=lambda a: a[-4:]))))
    observable_targets.append(seventypercent_targets)

get_ipython().magic('matplotlib inline')
from astroplan.plots import plot_airmass

target = 'SDSS083006.17+475150.29_15.81'

def sdss_name_to_target(name):
    ra = name[4:13]
    dec = name[13:-6]
    ra = "{0}h{1}m{2}s".format(ra[:2], ra[2:4], ra[4:])
    dec = "{0}d{1}m{2}s".format(dec[:3], dec[3:5], dec[5])
    return FixedTarget(SkyCoord(ra=ra, dec=dec), name=name)

present_time = Time.now()
if not obs.is_night(present_time):
    # If it's currently day time at runtime, find time of sunset and sunrise
    tonight_start = obs.twilight_evening_astronomical(present_time, which='next')
    tonight_end = obs.twilight_morning_astronomical(present_time, which='next')
else:
    # Otherwise find time to next sunrise
    tonight_start = present_time
    tonight_end = obs.twilight_morning_astronomical(present_time, which='next')
    
times = Time(np.linspace(tonight_start.jd, tonight_end.jd, 50), format='jd')
plot_airmass(sdss_name_to_target(target), obs, times);



