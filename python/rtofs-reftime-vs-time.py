get_ipython().magic('matplotlib inline')

import pandas as pd
from shapely.geometry import Point
import simplejson as json
from urllib.parse import urlencode
from urllib.request import urlopen, Request

apikey = 'ENTER_YOUR_API_KEY_HERE'

# Set the Planet OS API query parameters
count = 100
id = 'noaa_rtofs_surface_1h_diag'
lat=39.37858604638528
lon=-72.57739563685647
time_order = 'desc'

query_dict = {'apikey': apikey,
              'count': count,
              'lon': lon,
              'lat': lat,
              'time_order': time_order,
              }
query = urlencode(query_dict)

api_url = "http://api.planetos.com/v1/datasets/%s/point?%s" % (id, query)

request = Request(api_url)
response = urlopen(request)
response_json = json.loads(response.read())

data = response_json['entries']

# let's flatten the response and create a Pandas dataframe
df = pd.io.json.json_normalize(data)

# then index by time using the axes.time column
pd.to_datetime(df["axes.time"])
df.set_index('axes.time', inplace=True)

print(df.count())
df.head()

# Request only data from a specific model run

reftime_end = '2016-11-12T00:00:00'
reftime_start = '2016-11-12T00:00:00'

query_dict = {'apikey': apikey,
              'count': count,
              'lat': lat,
              'lon': lon,
              'reftime_end': reftime_end,
              'reftime_start': reftime_start,
              'time_order': time_order,
              }
query = urlencode(query_dict)

api_url = "http://api.planetos.com/v1/datasets/%s/point?%s" % (id, query)

request = Request(api_url)
response = urlopen(request)
response_json = json.loads(response.read())

data = response_json['entries']

# let's flatten the response and create a Pandas dataframe
df = pd.io.json.json_normalize(data)

# then index by time using the axes.time column
pd.to_datetime(df["axes.time"])
df.set_index('axes.time', inplace=True)

print(df.count())
df

# Request only data from yesterday's model run (e.g. 2016-11-11T00:00:00)

end = '2016-11-14T00:00:00'
start = '2016-11-14T00:00:00'

query_dict = {'apikey': apikey,
              'count': count,
              'end': end,
              'lat': lat,
              'lon': lon,
              'reftime_recent': 'false',
              'start': start,
              'time_order': time_order,
              }
query = urlencode(query_dict)

api_url = "http://api.planetos.com/v1/datasets/%s/point?%s" % (id, query)

request = Request(api_url)
response = urlopen(request)
response_json = json.loads(response.read())

data = response_json['entries']

# let's flatten the response and create a Pandas dataframe
df = pd.io.json.json_normalize(data)

# then index by time using the axes.time column
pd.to_datetime(df["axes.time"])
df.set_index('axes.time', inplace=True)

print(df.count())
df.head()



