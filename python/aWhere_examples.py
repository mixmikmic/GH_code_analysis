import datetime
import pandas as pd

from __future__ import division
from awhere import Awhere

DATE_FORMAT = '%Y-%m-%d'

# Setup key and sectret in aWhere.py
awhere = Awhere()

lat, lon = 10.01, 40.02
awhere.single_call(lat, lon, '2017-01-01', '2017-02-01')

lat, lon, title = 10.01, 40.02, 'Woreda1'
dataframe, failed = awhere.fetch_data_single(lat, lon, title, '2016-02-01', '2017-02-01')

#Write to CSV
dataframe.to_csv('data/test_single.csv')

lats = [1, 2, 3, 4]
lons = [-1, -2, -3, -4]
latlons = zip(lats, lons)
titles = ['geo1', 'geo2', 'geo3', 'geo4']

dataframe, failed = awhere.fetch_data_multiple(latlons, titles, '2016-02-01', '2017-02-01')

#Write to CSV
dataframe.to_csv('data/test_multiple.csv')

woreda_df = pd.read_csv('geo_locations/woreda_info.csv', dtype=str)
woreda_df = woreda_df[woreda_df['WoredaLat'].astype(float) != 0]

START_DATE_STR = '2016-01-01'
END_DATE = (datetime.datetime.now() - datetime.timedelta(days=1))
END_DATE_STR = END_DATE.strftime(DATE_FORMAT)

for lat, lon, geokey in woreda_df[['WoredaLat', 'WoredaLon', 'GeoKey']].values[:5]:
    
    dataframe, failures = awhere.fetch_data_single(lat, lon, geokey, START_DATE_STR, END_DATE_STR)
    if len(failures) > 0:
        print 'The following queries failed:' 
        print failures
    # Rename title as the GeoKey
    dataframe.rename(columns={'title': 'GeoKey'}, inplace=True)
    dataframe.to_csv('data/%s.csv' % geokey.replace(' ','_'))

woreda_dataframe = pd.read_csv('geo_locations/woreda_info.csv')

latlons = woreda_dataframe[['WoredaLat', 'WoredaLon']]
titles = woreda_dataframe['GeoKey']
latlons

dataframe, failed = awhere.fetch_data_multiple(latlons.values, titles.values, '2016-02-01', '2017-02-01')

dataframe.rename(columns={'title': 'GeoKey'}, inplace=True)
dataframe

