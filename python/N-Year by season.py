from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from datetime import datetime, timedelta, date
import operator
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
get_ipython().magic('matplotlib inline')

year_list = [year for year in range(1970, 2017)]

n_year_storms = pd.read_csv('data/n_year_storms_ohare_noaa.csv')
n_year_storms['start_time'] = pd.to_datetime(n_year_storms['start_time'])
n_year_storms['end_time'] = pd.to_datetime(n_year_storms['end_time'])
n_year_storms.head()

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

n_year_storms['season'] = n_year_storms['start_time'].apply(get_season)
n_year_storms.head()

# How often do N-Year Storms happen per season?
n_year_storms['season'].value_counts()

n_year_storms[n_year_storms['n'] >= 10]

n_year_storms[n_year_storms['n'] == 5]['season'].value_counts().plot(kind="pie", title='When 5-year storms happen')

n_year_storms[n_year_storms['n'] == 2]['season'].value_counts().plot(kind="pie", title='When 2-year storms happen')

n_year_storms[n_year_storms['n'] == 1]['season'].value_counts().plot(kind="pie", title='When 1-year storms happen')

smaller_storms = n_year_storms[n_year_storms['n'] <= 5]
smaller_storms_by_season = {year: {season: 0 for season in list(smaller_storms['season'].unique())} for year in year_list}
for index, storm in smaller_storms.iterrows():
    smaller_storms_by_season[storm['year']][storm['season']] += 1

smaller_storms_by_season = pd.DataFrame(smaller_storms_by_season).transpose()
smaller_storms_by_season.head()

smaller_storms_by_season.cumsum().plot(kind='line', title='Cumulative Number of 1-, 2-, and 5- year storms over time')

storms_by_season = {year: {season: 0 for season in list(n_year_storms['season'].unique())} for year in year_list}
for index, storm in n_year_storms.iterrows():
    storms_by_season[storm['year']][storm['season']] += 1

storms_by_season = pd.DataFrame(storms_by_season).transpose()
storms_by_season.head()

storms_by_season.cumsum().plot(kind='line', title='Cumulative Number of N-year storms over time')

# Curious about when we had the first n-year storm in each season
n_year_storms[n_year_storms['season'] != 'summer'].sort_values('year')

n_year_storms[n_year_storms['season'] == 'autumn']

storms_by_season.plot(kind='bar', stacked=True, title='N-Year Storms per Year by Season')



