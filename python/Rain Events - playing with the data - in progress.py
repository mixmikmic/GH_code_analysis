from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from datetime import datetime, timedelta, date
import operator
import matplotlib.pyplot as plt
from collections import namedtuple
get_ipython().magic('matplotlib notebook')

events = pd.read_csv('data/rain_events_ohare.csv')
events = events[(events['duration_hrs'] > 1) | (events['total_precip'] > 0.08)]
events['start_time'] = pd.to_datetime(events['start_time'])
events['end_time'] = pd.to_datetime(events['end_time'])
events.head()

events = events.set_index('start_time')
events['start_time'] = events.index.values
events['avg_intensity'] = events['total_precip'] / events['duration_hrs']
def find_year(timestamp):
    return timestamp.year
events['year'] = events['start_time'].apply(find_year)
events.head()

events['avg_intensity'].plot()

events[['year', 'avg_intensity']].groupby('year').mean().plot(kind='bar', title='Average event intensity by year')

events[['year', 'duration_hrs']].groupby('year').sum().plot(kind='bar', title='Total hours of rainfall per year')

events['year'].value_counts().sort_index().cumsum().plot(kind='bar', title='Cumulative number of events over time')

# Get the season of each event based on the start date
# This code is copied from http://stackoverflow.com/questions/16139306/determine-season-given-timestamp-in-python-using-datetime

Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

def get_season(timestamp):
    if isinstance(timestamp, datetime):
        timestamp = timestamp.date()
    timestamp = timestamp.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= timestamp <= end)

events['season'] = events['start_time'].apply(get_season)
events.head()

summer_events = events[events['season'] == 'summer']
summer_events.head()

summer_events['year'].value_counts().sort_index().plot(kind='bar', title='Number of summer events over time')

# TODO: Left off here









# Let's take a look at summer showers.  Are they happening with less frequencly over the years?
summer_events = events[events['season'] == 'summer']
summer_events.head()

(summer_events['year'].value_counts().sort_index()).plot(kind='bar')

# No!

events.plot()

summer_events[['year', 'avg_intensity']].groupby('year').mean()

summer_events[['year', 'avg_intensity']].groupby('year').mean().plot(kind='bar')









summer_events['hours_between_events'] = (summer_events['start_time'] - summer_events['end_time'].shift()).astype('timedelta64[h]')
summer_events.head()

# Average time between events - in hours
(summer_events.groupby('year')['hours_between_events'].mean()).plot(kind='bar')

# Events per season per year
per_season = {year: {'winter': 0, 'spring': 0, 'autumn': 0, 'summer': 0} for year in range(1970,2017)}
for index, event in events.iterrows():
    per_season[event['year']][event['season']] += 1
events_per_season_by_year = pd.DataFrame(per_season)
events_per_season_by_year = events_per_season_by_year.transpose()
events_per_season_by_year.head()

events_per_season_by_year['summer'].plot(kind='bar')



