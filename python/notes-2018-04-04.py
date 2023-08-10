import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

data = pd.read_csv('https://www.math.ubc.ca/~pwalls/data/weather.csv')

data.info()

data.head()

data = pd.read_csv('https://www.math.ubc.ca/~pwalls/data/weather.csv',usecols=[0,4,6,10])

data.head()

data.info()

first_row_date = data['date_time_local'][0]

print(first_row_date)

type(first_row_date)

int(first_row_date[0:4])

def get_year(text):
    return int(text[0:4])

get_year('2018-04-04 10:00:00 PDT')

data['year'] = data['date_time_local'].apply(get_year)

def get_month(text):
    return int(text[5:7])

def get_day(text):
    return int(text[8:10])

def get_hour(text):
    return int(text[11:13])

data['month'] = data['date_time_local'].apply(get_month)
data['day'] = data['date_time_local'].apply(get_day)
data['hour'] = data['date_time_local'].apply(get_hour)

data.head()

data.groupby('month')['temperature'].mean().plot(kind='bar')
plt.title('Average Monthly Temperature, 2012-2017')
plt.xlabel('Month')
plt.ylabel('Temperature (Celsius)')
plt.ylim([0,20])
plt.show()

temps = data.groupby(['month','year'])['temperature'].mean().unstack()

temps[[2013,2014,2015,2016,2017]].plot()
plt.title('Average Monthly Temperature by Year, 2013-2017')
plt.xlabel('Month')
plt.ylabel('Temperature (Celsius)')
plt.ylim([0,20])
plt.show()

data.groupby('wind_dir').size().plot(kind='bar')
plt.title('Total Wind Direction Measurements, 2012-2018')
plt.xlabel('Wind Direction')
plt.ylabel('Number of Measurements')
plt.show()

data.groupby('wind_dir')['wind_speed'].mean().plot(kind='bar')
plt.title('Average Wind Speed by Wind Direction, 2012-2018')
plt.xlabel('Wind Direction')
plt.ylabel('Wind Speed (km/h)')
plt.show()

