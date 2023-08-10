import numpy as np
import pandas as pd
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (12, 2)

import seaborn as sns
sns.set(font_scale=0.6)

import outlier
from utilities import *

df_hourly = pd.read_pickle("TrainValidationData/df_hourly.pkl")

print(list(df_hourly.Hour.cat.categories))
print(list(df_hourly.Night.cat.categories))
print(list(df_hourly.Station.cat.categories))

daily = df_hourly.reset_index().pivot_table(index = 'Date', columns = ['Station'], values='Exit', aggfunc=sum)
print("Number of days", len(daily))
for station in daily.columns:
    print("{:>22}: {:}".format(station, daily[station].isnull().sum()))

def missing_days(station, daily):
    for dt in daily[daily[station].isnull()].index:
        print(dt, dt.weekday_name)

missing_days('Kings Cross Station', daily)

missing_days('Parramatta Station', daily)

missing_days('Newtown Station', daily)

(df_hourly.reset_index().pivot_table(index = 'Date', columns = ['Station', 'Hour'], values='Exit', aggfunc=sum).
             isnull().
             sum().reset_index().
             pivot('Station', 'Hour', 0))

plot_stations_hourly(df_hourly,["Kings Cross Station", "Parramatta Station", 'Newtown Station'])

df_hourly_cleaned = (df_hourly.set_index(['Station', 'Night', 'Hour'], append=True).
                        groupby(level=['Station', 'Night', 'Hour'], as_index=False).
                        transform(outlier.replace, how='linear').
                        reset_index(level=['Station', 'Night', 'Hour']))
assert len(df_hourly) == len(df_hourly_cleaned), "Length of cleaned data is incorrect"

df_hourly_cleaned.to_pickle("TrainValidationData/df_hourly_cleaned.pkl")

df_daily = (df_hourly[df_hourly.Hour >= "7PM" ].
                reset_index().
                groupby(['Station', 'Night', 'Date'], as_index=False).
                agg({'Exit':sum, "Entry":sum}).
                set_index('Date'))

assert len(df_daily) == len(df_hourly)/len(df_hourly.Hour.cat.categories), "Wrong length"
assert df_daily.index.min() == df_hourly.index.min(), "Wrong start dates"
assert df_daily.index.max() == df_hourly.index.max(), "Wrong end dates"

df_daily.to_pickle("TrainValidationData/df_daily.pkl")

df_daily_cleaned = (df_hourly_cleaned[df_hourly_cleaned.Hour >= "7PM" ].
                        reset_index().
                        groupby(['Station', 'Night', 'Date'], as_index=False).
                        agg({'Exit':sum, "Entry":sum}).
                        set_index('Date'))

assert len(df_daily_cleaned) == len(df_hourly)/len(df_hourly.Hour.cat.categories), "Wrong length"

assert df_daily_cleaned.index.min() == df_hourly.index.min(), "Wrong start dates"
assert df_daily_cleaned.index.max() == df_hourly.index.max(), "Wrong end dates"

df_daily_cleaned.to_pickle("TrainValidationData/df_daily_cleaned.pkl")

plot_stations_daily(df_daily,["Kings Cross Station", "Parramatta Station", 'Newtown Station'])

plot_stations_daily(df_daily_cleaned,["Kings Cross Station", "Parramatta Station", 'Newtown Station'])

