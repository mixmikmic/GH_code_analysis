# Registered token
headers = {'token': 'CWvdeUOpiJMWYztUSajDbXWloJLoGBKR'}

# Obtain Available datasets
import requests

url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/datasets"
response = requests.get(url, headers = headers)
data_sets_response = response.json()


# Convert dataset into DataFrame
import pandas as pd
from pandas import DataFrame
pd.options.display.max_colwidth = 100

df = DataFrame.from_dict(data_sets_response['results'])
df.head(df.shape[0])

# Let's explore datatypes
url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes?limit=1000"
response = requests.get(url, headers = headers)
data_types_response = response.json()

df = DataFrame.from_dict(data_types_response['results'])

url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes?limit=1000&offset=1001"
response = requests.get(url, headers = headers)
data_types_response = response.json()

df2 = DataFrame.from_dict(data_types_response['results'])

df = pd.concat([df,df2]).reset_index()

pd.set_option('display.max_rows', df.shape[0])
df.head(df.shape[0])

# Let's explore datatypes
url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/datacategories?limit=1000"
response = requests.get(url, headers = headers)
data_categories_response = response.json()

df = DataFrame.from_dict(data_categories_response['results'])
df.head(df.shape[0])

# Let's explore location categories
url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/locationcategories"
response = requests.get(url, headers = headers)
location_categories_response = response.json()

df = DataFrame.from_dict(location_categories_response['results'])
df.head(df.shape[0])

# Explore Countries Available
url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/locations?locationcategoryid=CNTRY&limit=1000"
response = requests.get(url, headers = headers)
locations_response = response.json()

df = DataFrame.from_dict(locations_response['results'])
df.head(df.shape[0])

# Explore Climate Divisions Available
url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/locations?locationcategoryid=CLIM_DIV&limit=1000"
response = requests.get(url, headers = headers)
locations_response = response.json()

df = DataFrame.from_dict(locations_response['results'])
df.head(df.shape[0])

# Explore Climate Region Available
url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/locations?locationcategoryid=CLIM_REG&limit=1000"
response = requests.get(url, headers = headers)
locations_response = response.json()

df = DataFrame.from_dict(locations_response['results'])
df.head(df.shape[0])

# Explore the number of stations in the US
url = "http://www.ncdc.noaa.gov/cdo-web/api/v2/stations?locationcategoryid=FIPS:US"
response = requests.get(url, headers = headers)
locations_response = response.json()

print "Total Stations in the US: ", locations_response[u'metadata']['resultset']['count']

