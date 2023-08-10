import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

df = pd.read_csv("weatherstats_vancouver_hourly.csv")

df.head()

weather = pd.read_csv("weatherstats_vancouver_hourly.csv",usecols=[0,2,4,6,10],parse_dates=[0])

weather.head()

weather.info()

get_ipython().magic('pinfo pd.datetime')

right_now = pd.datetime(2017,3,31,2,15,0)

type(right_now)

right_now.year

right_now.day

weather['Year'] = weather['date_time_local'].map(lambda d : d.year)
weather['Month'] = weather['date_time_local'].map(lambda d : d.month)
weather['Day'] = weather['date_time_local'].map(lambda d : d.day)
weather['Hour'] = weather['date_time_local'].map(lambda d : d.hour)

weather.head()

weather.info()

weather.groupby('wind_dir').size().plot(kind='bar')

months = ['January','February','March','April','May','June','July','August','September','October','November','December']
plt.figure(figsize=(12,6))
plt.subplot(3,4,1)
for month in range(1,13):
    plt.subplot(3,4,month)
    wind_directions = weather[weather['Month'] == month].groupby('wind_dir').size()
    wind_directions = wind_directions / wind_directions.sum() * 100
    wind_directions.plot(kind='bar')
    plt.tight_layout()
    plt.title(months[month - 1]), plt.xlabel(''), plt.ylim([0,50])

months = ['January','February','March','April','May','June','July','August','September','October','November','December']
plt.figure(figsize=(12,6))
plt.subplot(3,4,1)
for month in range(1,13):
    plt.subplot(3,4,month)
    weather[weather['Month'] == month].groupby('Hour')['wind_speed'].mean().plot()
    plt.tight_layout()
    plt.title(months[month - 1]), plt.xlabel('Hour'), plt.ylabel('Wind speed (km/h)'), plt.ylim([0,20])

crime = pd.read_csv("van_crime.csv")

crime.head()

crime.info()

crime['TYPE'].unique()

crime[crime['TYPE'] == 'Homicide'].head()

crime[crime['X'] > 0].plot(kind='scatter',x='X',y='Y',alpha=0.2,s=2,figsize=(15,10))

