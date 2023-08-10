import numpy as np
import pandas as pd
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 2)

import seaborn as sns
sns.set(font_scale=0.6)

import outlier
import utilities as util

names=    ['Source', 'Line','Bands', 'Station', 'Entry', 'Exit', 'Date']
dtype=    {'Source':'category', 'Bands':'category', 'Station':'category', 'Entry':np.int32, 'Exit':np.int32}
usecols = ['Source', 'Station', 'Date', 'Bands',  'Entry', 'Exit']

df_raw = pd.concat([pd.read_csv(file, names=names, usecols=usecols, dtype=dtype, na_values=['', ' '], header=0)
                      for file in glob('TrainValidationData/*.zip')])

df_raw['Date'] = pd.to_datetime(df_raw.Date, format='%d/%m/%Y %H:%M')

BANDS = ["{} to {}".format(i,i+1) for i in range(0,27)]
df_raw.Bands = df_raw.Bands.astype('category', ordered=True,  categories=BANDS)

df_raw['Datetime'] = df_raw.Date + pd.to_timedelta(df_raw.Bands.cat.codes, unit='h')
df_raw['Date'] = pd.to_datetime((df_raw['Datetime'] - timedelta(hours = 6)).dt.date)

df_raw['Hour'] = df_raw.Datetime.dt.strftime('%I%p').astype('category', ordered=True, categories=['05PM', '06PM', '07PM', '08PM', '09PM', '10PM', '11PM', '12AM','01AM', '02AM'])
df_raw['Hour'] = df_raw.Hour.cat.rename_categories(['5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM', '12AM','1AM', '2AM'])
df_raw['Night'] = (df_raw.Date.dt.weekday_name.
                          astype('category', ordered=True, 
                                 categories=['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday']))

df_hourly = df_raw.groupby(['Station', 'Night', 'Date', 'Hour'], as_index=False).agg({'Entry':'sum', 'Exit':'sum'})

def check(new, old):
    
    print("Raw Lengths: ", len(new), len(old))
    print("Non null lengths", len(new.dropna()), len(old.dropna()))
    
    assert new.Exit.sum() == old.Exit.sum(), "Exit totals don't match"
    assert new.Entry.sum() == old.Entry.sum(), "Exit totals don't match"
    
    new = new.groupby(['Station',  'Date', 'Hour'], as_index=False).agg({'Entry':'sum', 'Exit':'sum'})
    old = new.groupby(['Station',  'Date', 'Hour'], as_index=False).agg({'Entry':'sum', 'Exit':'sum'})
    assert  len(new) == len(old), "Number of observations are different"
    

check(df_hourly, df_raw)

idx_levels = df_raw.set_index(['Station','Hour', 'Date']).index.levels
df_hourly = df_hourly.set_index(['Station','Hour', 'Date'])

idx = pd.MultiIndex.from_product(idx_levels, names=['Station','Hour', 'Date'])
df_hourly = df_hourly.reindex(index=idx).reset_index()

assert df_raw.Date.min() == df_hourly.Date.min(), "Wrong start dates"
assert df_raw.Date.max() == df_hourly.Date.max(), "Wrong end dates"

df_hourly['Night'] = (df_hourly.Date.dt.weekday_name.
                          astype('category', ordered=True, 
                                 categories=['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday']))

check(df_hourly, df_raw)

assert (df_hourly.Date.dt.weekday_name.unique() == df_hourly.Night.cat.categories).all(), "missing Night categories"
assert (df_hourly.Hour.unique() == df_hourly.Hour.cat.categories).all(), "missing Hour categories"
assert (df_hourly.Station.unique() == df_hourly.Station.cat.categories).all(), "missing Station categories"

df_hourly= df_hourly.query('Night in ["Friday", "Saturday", "Sunday"]')
df_hourly.Night = df_hourly.Night.cat.remove_unused_categories()

df_raw = df_raw.query('Night in ["Friday", "Saturday", "Sunday"]')
df_raw.Night = df_raw.Night.cat.remove_unused_categories()

print(df_raw.Date.min().date(), df_raw.Date.max().date()) 

df_raw.set_index('Date').to_pickle("TrainValidationData/df_raw.pkl")
df_hourly.set_index('Date').to_pickle("TrainValidationData/df_hourly.pkl")

