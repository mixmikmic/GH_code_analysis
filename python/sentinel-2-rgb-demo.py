get_ipython().magic('matplotlib inline')

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import dateutil.parser
from shapely.geometry import Point
import simplejson as json
from urllib.parse import urlencode
from urllib.request import urlopen, Request

apikey = 'YOUR-API-KEY-GOES-HERE'

def api_query(url):
#     print("API Url: %s" % url) # debug helper: display API urls in output
    request = Request(url)
    response = urlopen(request)
    return json.loads(response.read())

def api_dataset_query(id, params):
    '''
    Queries the Planet OS API dataset endpoint and returns the
    response as a JSON object.
    id (str): Planet OS dataset ID
    params (dict): Dict of API query parameters
    '''
    query = urlencode(params)
    url = "http://api.planetos.com/v1/datasets/%s?%s" % (id, query)
    return api_query(url)

def api_point_query(id, params):
    '''
    Queries the Planet OS API point endpoint and returns the
    response as a JSON object.
    
    id (str): Planet OS dataset ID
    params (dict): Dict of API query parameters
    '''
    query = urlencode(params)
    url = "http://api.planetos.com/v1/datasets/%s/point?%s" % (id, query)
    return api_query(url)

# The Planet OS dataset ids are required in the query string.
# Dataset IDs can be found on the dataset detail pages in the right-hand
# data access column once a user has authenticated.
ds_rgb_id = 'sentinel2_kenya_clouds_rgb'
ds_ndvi_id = 'sentinel2_kenya_clouds_ndvi'

# All queries require a valid API key
params = {'apikey': apikey, }

# Query the dataset endpoint for both RGB and NDVI
ds_rgb_json = api_dataset_query(ds_rgb_id, params)
ds_ndvi_json = api_dataset_query(ds_ndvi_id, params)

# Let's summarize the results in ASCII
def ds_summary(ds):
    '''
    Prints a summary of a dataset including title, variable names,
    variable long names, and unit.
    
    ds (json): Planet OS API dataset response in JSON format
    '''
    print(ds['Title'])
    print('-' * 80)
    print("{0:<30} {1:<40} {2:<10}".format("Variable", "Long Name", "Unit"))
    print('-' * 80)
    for v in ds['Variables']:
        name = v['name'] or '-'
        long_name =  v['longName'] or '-'
        unit = v['unit'] or '-'
        print("{0:<30} {1:<40} {2:<10}".format(name, long_name, unit))

ds_summary(ds_rgb_json)
print()
ds_summary(ds_ndvi_json)

# Select a point in decimal degrees to query. We'll use the centroid of a farm in Kenya.
# This particular point is interesting because it falls within 4 unique Sentinel 2
# grid tiles. As a result, our response will return values from each of the 4 tiles.
lon = 36.62117917585404
lat = -0.9584909201199898

params = {'apikey': apikey, # always required
          'count': 1, # number of values to return per classifier (e.g. tile)
          'lat': lat, # latitude of interest
          'lon': lon, # longitude of interest
          'max_count': 'true', # return total count of available values
          'nearest': 'true', # return data from the nearest available point
          'time_order': 'desc', # return data in descending chronological order
          'var': 'red,green,blue', # return red, green and blue variables
         }

ds_rgb_point = api_point_query(ds_rgb_id, params)
# print(json.dumps(ds_rgb_point, indent=2))

# The raw response contains two top level elements: 'entries' which contains the values
# and 'stats' which contains some metadata about the values. We'll create a Pandas
# dataframe with the values in 'entries'.

df = pd.io.json.json_normalize(ds_rgb_point['entries'])
print(df.count())
df.head()

# Let's increase the value count per tile to 5.
# Note that we could also use the maxCount value to acquire all available values as well.
# params['count'] = ds_rgb_point['stats']['maxCount']
params['count'] = 5

ds_rgb_point = api_point_query(ds_rgb_id, params)

df = pd.io.json.json_normalize(ds_rgb_point['entries'])
print(df.count())
df.head()

# drop NaN values and save as clean dataframe
df_clean = df.dropna()

# index by time using the axes.time column and sort descending
pd.to_datetime(df_clean["axes.time"])
df_clean.set_index('axes.time', inplace=True)
df_clean = df_clean.sort_index(ascending=False)

print(df_clean.describe())
df_clean.head()


params_compact = {'apikey': apikey, # always required
                  'buffer': 0.001, # return data from all points within 0.001 degree bounding box centered on lat/lon
                  'classifier:tile': '37MBU', # only return values from 37MBU tile
                  'count': 10, # number of values to return per classifier (e.g. tile)
                  'grouping': 'location', # compact into 2-d array by location axis (lat/lon)
                  'lat': lat, # latitude of interest
                  'lon': lon, # longitude of interest
                  'time_order': 'desc', # return data in descending chronological order
                  'var': 'red,green,blue', # return red, green and blue variables
                 }
ds_rgb_point_compact = api_point_query(ds_rgb_id, params_compact)
print(json.dumps(ds_rgb_point_compact))

# Use the same params query, but update 'var' to request cloud percentage
params['var'] = 'cloudy_pixels_percentage'
print(params, '\n') # output to refresh our memory

# Request the data and store in a dataframe
ds_rgb_clouds = api_point_query(ds_rgb_id, params)
df_clouds = pd.io.json.json_normalize(ds_rgb_clouds['entries'])

print(df_clouds.count())
df_clouds.head()

# Index rgb dataframe by time and tile, sort in descending order
df_tt = df.set_index(['axes.time','classifiers.tile']).sort_index(ascending=False)
print(df_tt.count())
df_tt.head()

# Index cloud coverage dataframe by time and tile, sort in descending order
df_clouds_tt = df_clouds.set_index(['axes.time','classifiers.tile']).sort_index(ascending=False)
print(df_clouds_tt.count())
df_clouds_tt.head()

# Concatenate the RGB and cloud percentage dataframes using an inner join 
df_rgbc = pd.concat([df_tt, df_clouds_tt], axis=1, join='inner')

# Drop the context columns to clean up the dataframe
df_rgbc.drop(['context'], axis=1, inplace=True)

# Sort by descending time
df_rgbc.sort_index(ascending=False, inplace=True)

print(df_rgbc.count())
df_rgbc.head(10)

df_rgbc_clean = df_rgbc.dropna()
print(df_rgbc_clean.count())
df_rgbc_clean.head()

# We can also use groupby to determine statistics over all available timestamps within each unique point.
df_rgbc_clean.groupby(['axes.latitude','axes.longitude']).describe()

