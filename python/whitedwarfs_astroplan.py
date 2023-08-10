from astroplan import Observer, FixedTarget
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

# Targets must be spectral typed and classified "Z", must be bright V<13
max_V = 14

def query_holberg(sptype_contains=None):
    catalog_name = "J/AJ/135/1225" #Holberg 2008"

    catalog_list = Vizier.find_catalogs(catalog_name)
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    Vizier.ROW_LIMIT = -1   # Otherwise would only show first 50 values
    catalog_table = catalogs[0] # This is the table with the data

    non_binaries = np.array([len(binarity) == 0 for binarity in catalog_table['Bin']])

    RAs = u.Quantity(catalog_table['_RAJ2000'].data[non_binaries], unit=u.deg)
    Decs = u.Quantity(catalog_table['_DEJ2000'].data[non_binaries], unit=u.deg)
    names = list(catalog_table['SimbadName'].data)
    sptypes = catalog_table['SpType'].data
    V_mags = catalog_table['Vmag'].data < max_V

    if sptype_contains is None:
        sptype_contains = ''
    
    holberg = [FixedTarget(coord=SkyCoord(ra=ra, dec=dec), name=name)
                 for ra, dec, name, sptype, V_mag in zip(RAs, Decs, names, sptypes, V_mags)
                 if sptype_contains.lower() in sptype.lower() and V_mag]
    holberg_V_mags = V_mags
    return holberg
    
holberg = query_holberg()
holberg_z = query_holberg('z')
print('len(holberg)={}'.format(len(holberg)))

def query_mccook(sptype_contains=None):

    catalog_name = "III/235B/"

    catalog_list = Vizier.find_catalogs(catalog_name)
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    Vizier.ROW_LIMIT = -1   # Otherwise would only show first 50 values
    catalog_table = catalogs[0] # This is the table with the data

    non_binaries = np.array([len(binarity) == 0 for binarity in catalog_table['bNote'].data])
    V_mags = catalog_table['Vmag'].data < max_V

    RAs = u.Quantity(catalog_table['_RAJ2000'].data[non_binaries], unit=u.deg)
    Decs = u.Quantity(catalog_table['_DEJ2000'].data[non_binaries], unit=u.deg)
    
    if sptype_contains is None:
        sptype_contains = ''
    
    mccook = [FixedTarget(coord=SkyCoord(ra=ra, dec=dec), name=name)
              for ra, dec, name, sptype, V_mag in zip(RAs, Decs, names, sptypes, V_mags)
              if sptype_contains in sptype.lower() and V_mag]
    mccook_V_mags = V_mags
    return mccook

mccook = query_mccook()
mccook_z = query_mccook('z')
print('len(mccook)={}'.format(len(mccook)))

obs = Observer.at_site("APO", timezone='US/Mountain')
target_list = holberg + mccook
#target_list = holberg_z + mccook_z
V_mag_list = list(holberg_V_mags) + list(mccook_V_mags)

from astroplan import is_observable, observability_table, AltitudeConstraint, AtNightConstraint
from astropy.time import Time

constraints = [AltitudeConstraint(min=30*u.deg), 
               AtNightConstraint.twilight_astronomical()]

# Figure out when "tonight" is
# present_time = Time.now()
# if not obs.is_night(present_time):
#     # If it's currently day time at runtime, find time of sunset and sunrise
#     tonight_start = obs.twilight_evening_astronomical(present_time, which='next')
#     tonight_end = obs.twilight_morning_astronomical(present_time, which='next')
# else:
#     # Otherwise find time to next sunrise
#     tonight_start = present_time
#     tonight_end = obs.twilight_morning_astronomical(present_time, which='next')

days_range = Time('2015-09-23 19:00') + np.array([0, 30, 60, 90])*u.day
months = ['Sept', 'Oct', 'Nov', 'Dec']
observable_targets = []
for day in days_range:
    tonight_start = obs.twilight_evening_astronomical(day, which='next')
    tonight_end = obs.twilight_morning_astronomical(day, which='next')
    table = observability_table(constraints, obs, target_list, 
                                time_range=Time([tonight_start, tonight_end]))
    seventypercent = table['fraction of time observable'] > 0.7
    seventypercent_targets = table['target name'][seventypercent].data
    observable_targets.append(seventypercent_targets)

for month, t, in zip(months, observable_targets):
    print("{}: {}".format(month, ', '.join(list(t))))
#unpack_targets = set(reduce(list.__add__, [list(t) for t in all_targets]))
#print(unpack_targets)



