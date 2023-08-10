from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import operator
get_ipython().magic('matplotlib inline')

# Get River reversals
reversals = pd.read_csv('data/lake_michigan_reversals.csv')
reversals['start_date'] = pd.to_datetime(reversals['start_date'])
reversals.head()

# Create rainfall dataframe.  Create a series that has hourly precipitation
rain_df = pd.read_csv('data/ohare_hourly_20160929.csv')
rain_df['datetime'] = pd.to_datetime(rain_df['datetime'])
rain_df = rain_df.set_index(pd.DatetimeIndex(rain_df['datetime']))
rain_df = rain_df['19700101':]
chi_rain_series = rain_df['HOURLYPrecip'].resample('1H', label='right').max()
chi_rain_series.head()

# Find the rainfall 'hours' hours before the timestamp
def cum_rain(timestamp, hours):
    end_of_day = (timestamp + timedelta(days=1)).replace(hour=0, minute=0)
    start_time = end_of_day - timedelta(hours=(hours-1))
    return chi_rain_series[start_time:end_of_day].sum()
    
t = pd.to_datetime('2015-06-15')
cum_rain(t, 240)

# Set the ten_day_rain field in reversals to the amount of rain that fell the previous 10 days (including the day that
# the lock was opened)
# TODO: Is there a more Pandaic way to do this?
for index, reversal in reversals.iterrows():
    reversals.loc[index,'ten_day_rain'] = cum_rain(reversal['start_date'], 240)
reversals

# Information about the 10 days that preceed these overflows
reversals['ten_day_rain'].describe(percentiles=[.25, .5, .75])

# N-Year Storm stuff
n_year_threshes = pd.read_csv('../../n-year/notebooks/data/n_year_definitions.csv')
n_year_threshes = n_year_threshes.set_index('Duration')
dur_str_to_hours = {
    '5-min':5/60.0,
    '10-min':10/60.0,
    '15-min':15/60.0,
    '30-min':0.5,
    '1-hr':1.0,
    '2-hr':2.0,
    '3-hr':3.0,
    '6-hr':6.0,
    '12-hr':12.0,
    '18-hr':18.0,
    '24-hr':24.0,
    '48-hr':48.0,
    '72-hr':72.0,
    '5-day':5*24.0,
    '10-day':10*24.0
}
n_s = [int(x.replace('-year','')) for x in reversed(list(n_year_threshes.columns.values))]
duration_strs = sorted(dur_str_to_hours.items(), key=operator.itemgetter(1), reverse=False)
n_year_threshes

# This method returns the first n-year storm found in a given interval.  It starts at the 100-year storm and decriments, so
# will return the highest n-year storm found
def find_n_year_storm(start_time, end_time):
    for n in n_s:
        n_index = n_s.index(n)
        next_n = n_s[n_index-1] if n_index != 0 else None

        for duration_tuple in reversed(duration_strs):

            duration_str = duration_tuple[0]
            low_thresh = n_year_threshes.loc[duration_str, str(n) + '-year']
            high_thresh = n_year_threshes.loc[duration_str, str(next_n) + '-year'] if next_n is not None else None
        
            duration = int(dur_str_to_hours[duration_str])
            sub_series = chi_rain_series[start_time: end_time]
            rolling = sub_series.rolling(window=int(duration), min_periods=0).sum()
        
            if high_thresh is not None:
                event_endtimes = rolling[(rolling >= low_thresh) & (rolling < high_thresh)].sort_values(ascending=False)
            else:
                event_endtimes = rolling[(rolling >= low_thresh)].sort_values(ascending=False)
            if len(event_endtimes) > 0:
                return {'inches': event_endtimes[0], 'n': n, 'end_time': event_endtimes.index[0], 'hours': duration}
    return None

start_time = pd.to_datetime('2008-09-04 01:00:00')
end_time = pd.to_datetime('2008-09-14 20:00:00')
find_n_year_storm(start_time, end_time)

# Add a column to the reversals data frame to show n-year storms that occurred before the reversal
# TODO: Is there a more Pandaic way to do this?
for index, reversal in reversals.iterrows():
    end_of_day = (reversal['start_date'] + timedelta(days=1)).replace(hour=0, minute=0)
    start_time = end_of_day - timedelta(days=10)
    reversals.loc[index,'find_n_year_storm'] = str(find_n_year_storm(start_time, end_of_day))
reversals

no_n_year = reversals.loc[reversals['find_n_year_storm'] == 'None']
print("There are %s reversals without an n-year event" % len(no_n_year))
no_n_year

reversals.loc[reversals['year'] == 1997]

reversals.sort_values('crcw', ascending=False)



