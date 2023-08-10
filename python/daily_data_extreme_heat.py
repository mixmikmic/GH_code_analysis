import requests 
import numpy as np
import pandas as pd
from datetime import datetime

# Convert value from degrees Celsius to degrees Fahrenheit (used for observed data)
def celsius_to_F(val):
    return val * 9/5 + 32 

# Convert value from Kelvin to degrees Fahrenheit (used for modeled data)
def kelvin_to_F(val):
    return  (val - 273.15) * 9/5 + 32

# Request header
headers = {'ContentType': 'json'}

# Uncomment the following lines to get data for a point location
#point = 'POINT(-121.4687 38.5938)'
#params = {'g': point}

# Uncomment the following lines to get data for a polygon
#polygon = 'POLYGON ((-123.35449 39.09596, -122.27783 39.09596, -122.27783 39.97712, -123.35449 39.97712, -123.35449 39.09596))' 
#params = {'g': polygon, 'stat': 'mean'}

# Your point of interest
point = 'POINT(-121.4687 38.5938)'
# Name of boundary layer in API
resource = 'counties'
# Request url
url = 'http://api.cal-adapt.org/api/%s/' % resource
# Request params to find intersecting boundaries
params = {'intersects': point, 'srs': 4326, 'simplify': .0001, 'precision': 4}
ref = ''

# Get geometry
response = requests.get(url, params=params, headers=headers)
if response.ok:
    data = response.json()
    feature = data['features'][0]
    if (feature):
        ref = '/api/%s/%s/' % (resource, feature['id'])
        print(ref)
    else:
        print('Did not find any polygons that intersect your point')

params = {'ref': ref, 'stat': 'mean'}

# Request url
url = 'http://api.cal-adapt.org/api/series/tasmax_day_livneh/' + 'rasters/'

# Make request
response = requests.get(url, params=params, headers=headers)

# Variable stores observed daily data in a Pandas dataframe
observedDF = None

if response.ok:
    json = response.json()
    data = json['results'][0]
    
    # Multiband raster data is returned by the API as a 3D array having a shape (233376, 1, 1)
    # Flatten the 3D array into a 1D array
    values_arr = np.array(data['image'])
    values_arr = values_arr.flatten()
    
    # Get start date of timeseries
    start_date = datetime.strptime(data['event'], '%Y-%m-%d')
    
    # Get total number of values -> number of days
    length = len(values_arr)
    
    # Create new pandas dataframe and map each value in list to a date index
    observedDF = pd.DataFrame(values_arr,
        index=pd.date_range(start_date, freq='1D', periods=length),
        columns=['value'])
    
    # Convert celsius to Fahrenheit
    observedDF.value = observedDF.value.apply(lambda x: celsius_to_F(x))

print(observedDF.head())
print()
print(observedDF.tail())

# Filter years
baselineDF = observedDF.loc['1961-01-01':'1990-12-31']

# Filter months
baselineDF = baselineDF[(baselineDF.index.month >= 4) & (baselineDF.index.month <= 10)]

print(baselineDF.head())
print()
print(baselineDF.tail())

threshold = baselineDF['value'].quantile(0.98, interpolation='linear')
print('Extreme Heat Threshold value is', round(threshold, 1), 'degrees Fahrenheit', sep = ' ')

# Request url
url = 'http://api.cal-adapt.org/api/series/tasmax_day_HadGEM2-ES_rcp85/' + 'rasters/'

# Make request
response = requests.get(url, params=params, headers=headers)

# Variable stores modeled daily data in a Pandas dataframe
modeledDF = None

if response.ok:
    json = response.json()
    data = json['results'][0]
    
    # Multiband raster data is returned by the API as a 3D array having a shape (233376, 1, 1)
    # Flatten the 3D array into a 1D array
    values_arr = np.array(data['image'])
    values_arr = values_arr.flatten()
    
    # Get start date of timeseries
    start_date = datetime.strptime(data['event'], '%Y-%m-%d')
    
    # Get total number of values -> number of days
    length = len(values_arr)
    
    # Create new pandas dataframe and map each value in list to a date index
    modeledDF = pd.DataFrame(values_arr,
        index=pd.date_range(start_date, freq='1D', periods=length),
        columns=['value'])
    
    # Convert Kelvin to Fahrenheit
    modeledDF.value = modeledDF.value.apply(lambda x: kelvin_to_F(x))

print(modeledDF.head())
print()
print(modeledDF.tail())

# Filter years
filteredDF = modeledDF.loc['2070-01-01':'2099-12-31']

# Filter months
filteredDF = filteredDF[(filteredDF.index.month >= 4) & (filteredDF.index.month <= 10)]

# Filter days > threshold
filteredDF = filteredDF[filteredDF.value > threshold]

print(filteredDF.head())
print()
print(filteredDF.tail())

filteredDF.value.resample('1AS').count()

filteredDF.value.resample('1AS').count().mean()

