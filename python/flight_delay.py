import pandas as pd
from dask import delayed
import numpy as np

# Define @delayed-function read_flights
@delayed
def read_flights(filename):

    # Read in the DataFrame: df
    df = pd.read_csv(filename,parse_dates=['FL_DATE'])

    # Calculate df['WEATHER_DELAY']
    df['WEATHER_DELAY'] = df['WEATHER_DELAY'].replace(0,np.nan)

    # Return df
    return df

filenames=['flightdelays-2016-1.csv',
 'flightdelays-2016-2.csv',
 'flightdelays-2016-3.csv',
 'flightdelays-2016-4.csv',
 'flightdelays-2016-5.csv',
 'flightdelays-2016-6.csv',
 'flightdelays-2016-7.csv',
 'flightdelays-2016-8.csv',
 'flightdelays-2016-9.csv',
 'flightdelays-2016-10.csv',
 'flightdelays-2016-11.csv',
 'flightdelays-2016-12.csv']

# Loop over filenames with index filename
for filename in filenames:
    # Apply read_flights to filename; append to dataframes
    dataframes.append(read_flights(filename))

# Compute flight delays: flight_delays
flight_delays = dd.from_delayed(dataframes)

# Print average of 'WEATHER_DELAY' column of flight_delays
print(flight_delays['WEATHER_DELAY'].mean().compute())

# Define @delayed-function read_weather with input filename
@delayed
def read_weather(filename):
    # Read in filename: df
    df = pd.read_csv(filename,parse_dates=['Date'])

    # Clean 'PrecipitationIn'
    df['PrecipitationIn'] = pd.to_numeric(df['PrecipitationIn'],errors='coerce')

    # Create the 'Airport' column
    df['Airport'] = filename.splied('.')[0]

    # Return df
    return df

# Loop over filenames with filename
for filename in filenames:
    # Invoke read_weather on filename; append result to weather_dfs
    weather_dfs.append(read_weather(filename))

# Call dd.from_delayed() with weather_dfs: weather
weather = dd.from_delayed(weather_dfs)

# Print result of weather.nlargest(1, 'Max TemperatureF')
print(weather.nlargest(1, 'Max TemperatureF').compute())

# Make cleaned Boolean Series from weather['Events']: is_snowy
is_snowy = weather['Events'].str.contains('Snow').fillna(False)

# Create filtered DataFrame with weather.loc & is_snowy: got_snow
got_snow = weather.loc[is_snowy]

# Groupby 'Airport' column; select 'PrecipitationIn'; aggregate sum(): result
result = got_snow.groupby('Airport')['PrecipitationIn'].sum()

# Compute & print the value of result
print(result.compute())



def percent_delayed(df):
    return (df['WEATHER_DELAY'].count() / len(df)) * 100



