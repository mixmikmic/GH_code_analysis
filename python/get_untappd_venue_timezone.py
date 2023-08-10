import time, requests, pytz, pandas as pd
from keys import google_timezone_api_key
from dateutil import parser as date_parser

# define pause interval to not hammer their server
pause = 0.1

# load the data, parse datetime string to datetime object, and combine lat-long into single column
df = pd.read_csv('data/untappd_details_geocoded.csv', encoding='utf-8')
df['date_pacific_tz'] = df['date_pacific_tz'].map(lambda x: date_parser.parse(x))
df['venue_latlng'] = df.apply(lambda row: '{},{}'.format(row['venue_lat'], row['venue_lon']), axis=1)
df.head()

# how many total venue lat-longs are there, and how many unique lat-longs are there?
print(len(df['venue_latlng']))

venue_latlngs_unique = pd.Series(df['venue_latlng'].unique())
print(len(venue_latlngs_unique))

venue_latlngs_unique = venue_latlngs_unique.sort_values()

# send each unique lat-long to the google timezone api to retrieve the local time zone id at that location
def get_timezone_google(latlng, timestamp=0):
    time.sleep(pause)
    url = 'https://maps.googleapis.com/maps/api/timezone/json?location={}&timestamp={}&key={}'
    request = url.format(latlng, timestamp, google_timezone_api_key)
    response = requests.get(request)
    data = response.json()
    try:
        return data['timeZoneId']
    except:
        return None
    
timezones = venue_latlngs_unique.map(get_timezone_google)

# create a dict with key of lat-long and value of timezone
latlng_timezone = {}
for label in timezones.index:
    key = venue_latlngs_unique[label]
    val = timezones[label]
    latlng_timezone[key] = val

# for each row in the df, look up the lat-long in the dict to get the local timezone
def get_timezone_from_dict(venue_latlng):
    try:
        return latlng_timezone[venue_latlng]
    except:
        return None

df['venue_timezone'] = df['venue_latlng'].map(get_timezone_from_dict)
df = df.drop('venue_latlng', axis=1)

# backfill timezones from the next earlier observation as this is more likely to be accurate...
# ...than randomly using the default timezone
df['venue_timezone'] = df['venue_timezone'].fillna(method='bfill')

# convert each row's datetime to the local timezone of the venue i checked into
def localize_date_time(row):
    date_time = row['date_pacific_tz']
    local_timezone = row['venue_timezone']
    try:
        return date_time.astimezone(pytz.timezone(local_timezone))
    except:
        return None
    
df['date_local_tz'] = df.apply(localize_date_time, axis=1)

# look at the first 10 venues and their timezones
df[['venue_name', 'venue_place', 'venue_timezone', 'date_pacific_tz', 'date_local_tz']].head(10)

# save to csv
df.to_csv('data/untappd_details_geocoded_timezone.csv', index=False, encoding='utf-8')



