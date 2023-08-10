get_ipython().magic('matplotlib inline')

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn  # make charts prettier and more readable

# Let's load one of the CSV to see what they look like
nice = pd.read_csv('nice.csv')
nice[:5]

_ = nice['avg_temp'].plot(figsize=(15, 5))

locations = ['nice', 'montreal', 'okinawa', 'london']
weather = pd.DataFrame()

for location in locations:
    frame = pd.read_csv('%s.csv' % location)
    # We need to keep track of where it's coming from obviously
    frame['location'] = location
    weather = weather.append(frame)

# Alternative to using slicing
weather.head()

weather.describe()

# Define some styles that we will reuse for all line graphs
styles = {
    'london': 'go-',
    'nice': 'ro-',
    'montreal': 'bo-',
    'okinawa': 'co-',
}

# we define a method since we will need to do that pretty often
def plot_grouped_by(dataframe, column_name):
    """Plots the dataframe grouped by location for the given column"""
    # Need to use the month as the index
    locations = dataframe.set_index('month').groupby('location')
    
    for loc_name, loc in locations:
        loc[column_name].plot(x='month', label=str(loc_name), style=styles[str(loc_name)])


plt.figure(figsize=(16, 8))
ax = plt.subplot(111)

plot_grouped_by(weather, 'avg_temp')

# Yes, I did add the 40 degrees tick just to be able to fit the legend properly
plt.yticks([-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40], fontsize=14)
plt.legend(fontsize=14, loc="upper left")
plt.title("Monthly average temperature 2011-2014", fontsize=16)
_ = plt.ylabel("Temperature (celsius)", fontsize=16) 
_ = plt.xlabel("Time", fontsize=16) 

# Making sure we have a datetime first rather than a string
weather['month'] = pd.to_datetime(weather['month'], format="%m-%Y")
start = datetime.date(2014, 1, 1)

# pandas allows all kind of iterator magic
weather_2014 = weather[weather.month >= start]
weather_2014.head()

# Let's look at temperatures and humidity
plt.figure(figsize=(16, 8))

# this 221 means we want a 2x2 plots display and this is the first one 
# (so upper left)
ax = plt.subplot(221)

plot_grouped_by(weather_2014, 'avg_temp')
plt.yticks([-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35], fontsize=14)
plt.legend(fontsize=14, loc="lower center")
plt.title("Monthly average temperature 2014")
_ = plt.ylabel("Temperature (celsius)", fontsize=16) 

ax2 = plt.subplot(222)

plot_grouped_by(weather_2014, 'max_temp')
plt.yticks([0, 10, 20, 30, 40], fontsize=14)
plt.legend(fontsize=14, loc="lower center")
plt.title("Max temperature 2014")
_ = plt.ylabel("Temperature (celsius)", fontsize=16) 

ax3 = plt.subplot(223)

plot_grouped_by(weather_2014, 'min_temp')
plt.yticks([-30, -20, -10, 0, 10, 20, 30], fontsize=14)
plt.legend(fontsize=14, loc="lower center")
plt.title("Min temperature 2014")
_ = plt.ylabel("Temperature (celsius)", fontsize=16) 

ax4 = plt.subplot(224)

plot_grouped_by(weather_2014, 'humidity')
plt.title("Average monthly humidity % in 2014 (legend identical)")
_ = plt.ylabel("Humidity %", fontsize=16) 


colors = {
    'london': 'green',
    'nice': 'red',
    'montreal': 'blue',
    'okinawa': 'cyan',
}

def bar_plot(ax, column_name):
    weather_2014.set_index(
        ['month', 'location']
    ).unstack().plot(
        ax=ax,
        kind='bar', 
        y=column_name
    )

plt.figure(figsize=(16, 8))
ax = plt.subplot(211)

bar_plot(ax, "raindays")
plt.legend(fontsize=14, loc="best")
plt.title("Rain days per month in 2014")
_ = plt.ylabel("Number of rain days", fontsize=16) 

ax2 = plt.subplot(212)

bar_plot(ax2, "snowdays")
plt.legend(fontsize=14, loc="best")
plt.title("Snow days per month in 2014")
_ = plt.ylabel("Number of snow days", fontsize=16) 

