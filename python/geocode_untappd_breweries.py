import requests, time, pandas as pd, numpy as np

# set the pause duration between api requests
pause = 0.1

# load the untappd data and look at the first 10 brewery names and places
df = pd.read_csv('data/untappd_details.csv', encoding='utf-8')
df[['brewery_name', 'brewery_place']].head(10)

# how many total brewery places are there, and how many unique places are there?
print(len(df['brewery_place']))
print(len(df['brewery_place'].unique()))

# first clean up any places with parentheses or slashes, like myanmar and china
df['brewery_place'] = df['brewery_place'].map(lambda x: x.split(' (')[0])
df['brewery_place'] = df['brewery_place'].map(lambda x: x.split(' /')[0])

# select only the unique places and drop any that are just 'other'
brewery_places = pd.Series(df['brewery_place'].unique())
brewery_places = brewery_places[~(brewery_places=='Other')]
brewery_places = brewery_places.sort_values()

# function that accepts an address string, sends it to the Google API, and returns the lat-long API result
def geocode_google(address):
    time.sleep(pause) #pause for some duration before each request, to not hammer their server
    url = 'http://maps.googleapis.com/maps/api/geocode/json?address={}&sensor=false' #api url with placeholders
    request = url.format(address)
    response = requests.get(request)
    data = response.json()
    
    if len(data['results']) > 0: #if google was able to geolocate our address, extract lat-long from result
        latitude = data['results'][0]['geometry']['location']['lat']
        longitude = data['results'][0]['geometry']['location']['lng']
        return '{},{}'.format(latitude, longitude) #return lat-long as a string in the format google likes

def geocode_nominatim(address):
    time.sleep(pause)
    url = 'https://nominatim.openstreetmap.org/search?format=json&q={}'
    request = url.format(address)
    response = requests.get(request)
    data = response.json()
    if len(data) > 0:
        return '{},{}'.format(data[0]['lat'], data[0]['lon'])

# geocode all the unique brewery places using the google maps api
brewery_latlngs = brewery_places.map(geocode_google)

# how many places failed to geocode?
brewery_places_failed = brewery_places[brewery_latlngs[pd.isnull(brewery_latlngs)].index]
print('after google, {} places lack lat-long'.format(len(brewery_places_failed)))

# re-try any that failed to geocode, but this time use the nominatim api
brewery_latlngs_nominatim = brewery_places_failed.map(geocode_nominatim)
brewery_places_failed = brewery_places[brewery_latlngs_nominatim[pd.isnull(brewery_latlngs_nominatim)].index]
print('after nominatim, {} places lack lat-long'.format(len(brewery_places_failed)))

# update the latlng values in brewery_latlngs based on any new results from nominatim
for label in brewery_latlngs_nominatim.index:
    brewery_latlngs[label] = brewery_latlngs_nominatim[label]

# create a dict with key of place name and value of lat-long
place_latlng = {}
for label in brewery_places.index:
    key = brewery_places[label]
    val = brewery_latlngs[label]
    place_latlng[key] = val

def get_latlng(brewery_place):
    try:
        return place_latlng[brewery_place]
    except:
        return None

df['brewery_latlng'] = df['brewery_place'].map(get_latlng)

# split latlng into separate lat and lon columns
df['brewery_lat'] = df['brewery_latlng'].map(lambda x: x.split(',')[0] if pd.notnull(x) else np.nan)
df['brewery_lon'] = df['brewery_latlng'].map(lambda x: x.split(',')[1] if pd.notnull(x) else np.nan)
df = df.drop('brewery_latlng', axis=1)

# look at the first 10 breweries and their lat-longs
df[['brewery_name', 'brewery_place', 'brewery_lat', 'brewery_lon']].head(10)

# save to csv
df.to_csv('data/untappd_details_geocoded.csv', index=False, encoding='utf-8')



