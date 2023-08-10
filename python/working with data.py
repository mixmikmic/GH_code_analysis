# Hands-On With Python

import pandas as pd
import numpy as np

rentals_df = pd.read_csv('rentals.csv', encoding='latin-1')

rentals_df.head()

rentals_df.shape

print("There are {} records and {} columns in the rentals.csv file.".format(rentals_df.shape[0], rentals_df.shape[1]))

rentals_df.dtypes

rentals_df['Tripduration_mins'] = rentals_df['Tripduration'] / 60

rentals_df.tail()

# Parse the date time object from a string
rentals_df['Starttime_dt'] = pd.to_datetime(rentals_df['Starttime'], format='%m/%d/%Y %H:%M')

# Format the date time object as a string
rentals_df['Startdate'] = rentals_df['Starttime_dt'].dt.strftime('%m/%d/%Y')

rentals_df.head()

print(rentals_df['Starttime'][0]>rentals_df['Starttime'][1])

rentals_df.drop(['Trip id', 'Bikeid', 'Starttime', 'Stoptime', 'Tripduration',                  'From station id', 'To station id'],                 axis=1, inplace=True)

rentals_df.head()

june_rides = rentals_df[rentals_df['Starttime_dt'] > '2017-06-01']
june_rides.head()

june_start_dates = june_rides[['Starttime_dt']]
june_start_dates.head()

count_of_rides = june_start_dates.groupby(june_start_dates['Starttime_dt'].dt.date).count()
count_of_rides.head()

june_start_dates['Starttime_dt'].dt.dayofweek

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

count_of_rides.index.name = 'Ride Start Date'

count_of_rides.plot(kind="bar", title='Frequency of Healthy Rides (June 2017)',figsize=(15,4))

diff_station_rides=rentals_df[rentals_df['From station name']!=rentals_df['To station name']]
diff_station_rides.head()

diff_station_rides.shape

morethan100_rides=diff_station_rides.groupby(['From station name', 'To station name']).count()
morethan100_rides=morethan100_rides.drop(['Tripduration_mins','Starttime_dt','Startdate'],axis=1)
morethan100_rides.columns=['Ride counts']
morethan100_rides=morethan100_rides[morethan100_rides['Ride counts']>100]
morethan100_rides.head(15)

aggregated_rides=morethan100_rides.groupby(['From station name'])['Ride counts'].sum()
aggregated_rides.head()

aggregated_rides.plot(kind='bar',figsize=(12,4))

