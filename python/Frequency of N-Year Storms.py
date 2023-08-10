from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from datetime import datetime, timedelta
import operator
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
get_ipython().magic('matplotlib inline')

n_year_storms = pd.read_csv('data/n_year_storms_ohare_noaa.csv')
n_year_storms['start_time'] = pd.to_datetime(n_year_storms['start_time'])
n_year_storms['end_time'] = pd.to_datetime(n_year_storms['end_time'])
n_year_storms = n_year_storms.set_index('start_time')
n_year_storms.head()

# Based on previous notebooks, we should have 83 n-year events in this timeframe.
len(n_year_storms)

ns_by_year = {year: {n: 0 for n in list(n_year_storms['n'].unique())} for year in range(1970, 2017)}
for index, event in n_year_storms.iterrows():
    ns_by_year[event['year']][int(event['n'])] += 1
ns_by_year = pd.DataFrame(ns_by_year).transpose()
ns_by_year.head()

# Double check that we still have 83 events
ns_by_year.sum().sum()

all_years = [i for i in range(1970, 2016)]
small_events = ns_by_year[(ns_by_year[1] > 0) | (ns_by_year[2] > 0)][[1,2]]
small_events = small_events.reindex(all_years, fill_value=0)
small_events.columns = [str(n) + '-year' for n in small_events.columns]
small_events.head()

# Number of 1 and 2 year events per year
small_events.cumsum().plot(kind='line', stacked=False, title="1- and 2-year Storms by Year - Cumulative Total over Time")

# Divide into buckets using resampling
n_year_storms.resample('15A',how={'year':'count'})

# Using the resample method is not really want giving me what I want.  Do this brute force
# TODO: Play around with resample to do this more efficiently

# I'd like to try and be a little more explicit in how I'm breaking this up
def find_bucket(year):
    if year < 1986:
        return '1970-1985'
    elif year <= 2000:
        return '1986-2000'
    else:
        return '2001-2015'
ns_by_year['year'] = ns_by_year.index.values
ns_by_year['bucket3'] = ns_by_year['year'].apply(find_bucket)
ns_by_year = ns_by_year.drop('year', 1)
ns_by_year.head()

bucket3 = ns_by_year.groupby('bucket3').sum()
bucket3.head()

# Make sure there are 83 storms
bucket3.sum().sum().sum()

bucket3.plot(kind='bar', stacked=True, title="N-Year Storms across 3 time intervals")

ns_by_year.head()

def find_bucket(year):
    if year < 1976:
        return "1970-1975"
    elif year < 1981:
        return '1976-1980'
    elif year < 1986:
        return '1981-1985'
    elif year < 1991:
        return '1986-1990'
    elif year < 1996:
        return '1991-1995'
    elif year < 2001:
        return '1996-2000'
    elif year < 2006:
        return '2001-2005'
    elif year < 2011:
        return '2006-2010'
    else:
        return '2011-2015'
ns_by_year['year'] = ns_by_year.index.values
ns_by_year['bucket8'] = ns_by_year['year'].apply(find_bucket)
ns_by_year = ns_by_year.drop('year', 1)
ns_by_year.head()

bucket8 = ns_by_year.drop('bucket3',1).groupby('bucket8').sum()
bucket8.head()

bucket8.sum().sum().sum()

bucket8.plot(kind='bar', stacked=True, title="N-Year Storms across 8 Intervals")



