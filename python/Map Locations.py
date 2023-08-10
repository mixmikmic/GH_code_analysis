# Click the Blue Plane to preview this notebook as a CrossCompute Tool
target_folder = '/tmp/map-locations'
location_table_path = 'usa-height-for-3-waterfalls.csv'

from os.path import join
from pandas import read_csv
location_table = read_csv(location_table_path)
location_table

import geopy
geocode = geopy.GoogleV3().geocode
geocode('New York, NY')

latitudes = []
longitudes = []
for name in location_table['Name']:
    location = geocode(name, timeout=5)
    latitudes.append(location.latitude)
    longitudes.append(location.longitude)
    print(name)

location_table['Latitude'] = latitudes
location_table['Longitude'] = longitudes
location_table

from invisibleroads_macros.disk import make_folder
from os.path import join
target_path = join(make_folder(target_folder), 'locations.csv')
location_table.to_csv(target_path, index=False)
print('location_geotable_path = %s' % target_path)

