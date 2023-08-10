from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().magic('matplotlib inline')

root_proj_dir = os.path.dirname(os.getcwd())
n_year_data_dir = os.path.join(root_proj_dir, 'n-year', 'notebooks', 'data')

rain_df = pd.read_csv(os.path.join(n_year_data_dir, 'ohare_full_precip_hourly.csv'))
rain_df['DATE'] = pd.to_datetime(rain_df['DATE'])
rain_df = rain_df.set_index(pd.DatetimeIndex(rain_df['DATE']))
print(rain_df.shape)
print(rain_df.dtypes)
rain_df.head()

cso_df = pd.read_csv('data/merged_cso_data.csv')
cso_df['Open date/time'] = pd.to_datetime(cso_df['Open date/time'])
cso_df['Gate Open Period'] = pd.to_timedelta(cso_df['Gate Open Period'], unit='m')
cso_df['Open Period Seconds'] = cso_df['Gate Open Period'].astype('timedelta64[s]')
cso_df = cso_df.set_index(pd.DatetimeIndex(cso_df['Open date/time']))
print(cso_df.shape)
print(cso_df.dtypes)
cso_df.head()

# Getting earliest dates to pull weather for
cso_df.sort_values(by='Open date/time').head()

# Pulling from April of 2009 to get some additional padding before earliest CSO
cso_rain_df = rain_df['2009-04-01':]
cso_rain_df.head()

# Get daily precipitation through sum of values, plot daily precipitation
cso_rain_series = cso_rain_df['HOURLYPrecip'].resample('1D').sum()
cso_rain_series.plot()

cso_df['Date'] = cso_df.index.date
print(cso_df.dtypes)
cso_df.head()

# Group CSO events by date, plot top 10
# Looking just for number of events, 
cso_by_date = cso_df.groupby(['Date'])['Open Period Seconds'].sum()
cso_date_sub = cso_by_date.sort_values(ascending=False)
cso_date_sub[:10].plot(kind='bar')

# June 15 of 2015 seems to have been the most severe CSO event, looking at what it consisted of
june_15_cso = cso_df['2015-06-15']
june_15_cso.head()

longest_june_15 = june_15_cso.sort_values(by='Open Period Seconds', ascending=False)
longest_june_15[:10].plot(kind='bar', x='Outfall Structure', y='Open Period Seconds')

# Looking at Outfall Structures ranked by total length of CSOs
cso_by_structure = cso_df.groupby(['Outfall Structure'])['Open Period Seconds'].sum()
cso_by_structure = cso_by_structure.sort_values(ascending=False)
cso_by_structure[:10].plot(kind='bar')

# Merge top CSO days by length of open gates in CSOs with total precipitation on that day
cso_date_df = pd.DataFrame(cso_by_date).reset_index()
cso_date_df['Date'] = pd.to_datetime(cso_date_df['Date'])
cso_date_df = cso_date_df.set_index(pd.DatetimeIndex(cso_date_df['Date']))
cso_date_df['Precipitation'] = cso_rain_series
print(cso_date_df.shape)
print(cso_date_df.dtypes)
cso_date_df.head()

# Plotting all event open period length by precipitation, need to reduce to just top events
cso_date_df.plot(kind='scatter', x='Precipitation', y='Open Period Seconds')

cso_date_df = cso_date_df.sort_values(by='Open Period Seconds', ascending=False)
# Plot top 100 CSO dates in terms of gate open period by precipitation on that date
cso_date_df[:100].plot(kind='scatter', x='Precipitation', y='Open Period Seconds')

# Ignoring top event because skews the results significantly
cso_date_df[1:100].plot(kind='scatter', x='Precipitation', y='Open Period Seconds')

cso_rain_2wk = cso_rain_series.rolling(window=14).sum()

cso_date_2wk = pd.DataFrame(cso_by_date).reset_index()
cso_date_2wk['Date'] = pd.to_datetime(cso_date_2wk['Date'])
cso_date_2wk = cso_date_2wk.set_index(pd.DatetimeIndex(cso_date_2wk['Date']))
cso_date_2wk['TwoWeekPrecip'] = cso_rain_2wk

print(cso_date_2wk.shape)
print(cso_date_2wk.dtypes)
cso_date_2wk.head()

cso_date_2wk.plot(kind='scatter', x='TwoWeekPrecip', y='Open Period Seconds')

cso_date_2wk = cso_date_2wk.sort_values(by='Open Period Seconds', ascending=False)
# Plot top 100 CSO dates in terms of gate open period by accumulated 2 week precipitation at that date
cso_date_2wk[:100].plot(kind='scatter', x='TwoWeekPrecip', y='Open Period Seconds')

# Ignore outlier
cso_date_2wk[1:100].plot(kind='scatter', x='TwoWeekPrecip', y='Open Period Seconds')

# May be more of a relationship in the two week window, moving to a month
cso_rain_month = cso_rain_series.rolling(window=30).sum()

cso_date_month = pd.DataFrame(cso_by_date).reset_index()
cso_date_month['Date'] = pd.to_datetime(cso_date_month['Date'])
cso_date_month = cso_date_month.set_index(pd.DatetimeIndex(cso_date_month['Date']))
cso_date_month['MonthPrecip'] = cso_rain_month

print(cso_date_month.shape)
print(cso_date_month.dtypes)
cso_date_month.head()

cso_date_month.plot(kind='scatter', x='MonthPrecip', y='Open Period Seconds')

cso_date_month = cso_date_month.sort_values(by='Open Period Seconds', ascending=False)
# Plot top 100 CSO dates in terms of gate open period by accumulated monthly precipitation at that date
cso_date_month[:100].plot(kind='scatter', x='MonthPrecip', y='Open Period Seconds')

# Outlier removed
cso_date_month[1:100].plot(kind='scatter', x='MonthPrecip', y='Open Period Seconds')

# Multiple outliers factor in here, see what plot looks like without top 5
cso_date_month[5:100].plot(kind='scatter', x='MonthPrecip', y='Open Period Seconds')



