from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from datetime import datetime, timedelta
import operator
import matplotlib.pyplot as plt
from collections import namedtuple
get_ipython().magic('matplotlib notebook')

# The following code is adopted from Pat's Rolling Rain N-Year Threshold.pynb
# Loading in hourly rain data from CSV, parsing the timestamp, and adding it as an index so it's more useful

rain_df = pd.read_csv('data/ohare_hourly_20160929.csv')
rain_df['datetime'] = pd.to_datetime(rain_df['datetime'])
rain_df = rain_df.set_index(pd.DatetimeIndex(rain_df['datetime']))
rain_df = rain_df['19700101':]
chi_rain_series = rain_df['HOURLYPrecip'].resample('1H', label='right').max().fillna(0)
chi_rain_series.head()

# N-Year Storm variables
# These define the thresholds laid out by bulletin 70, and transfer mins and days to hours
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

# Roll through the rain series in intervals at the various durations, and look for periods that exceed the thresholds above.
# Return a DataFrame
def find_n_year_storms(start_time, end_time, n):
    n_index = n_s.index(n)
    next_n = n_s[n_index-1] if n_index != 0 else None
    storms = []

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
        for index, event_endtime in event_endtimes.iteritems():
            storms.append({'n': n, 'end_time': index, 'inches': event_endtime, 'duration_hrs': duration,
                          'start_time': index - timedelta(hours=duration)})
    return pd.DataFrame(storms)

# Find all of the n-year storms in the whole rainfall dataset
n_year_storms_raw = find_n_year_storms(chi_rain_series.index[0], chi_rain_series.index[-1], 100)
for n in n_s[1:]:
    n_year_storms_raw = n_year_storms_raw.append(find_n_year_storms(chi_rain_series.index[0], chi_rain_series.index[-1], n))
n_year_storms_raw.head()

# Re-order the dataframe to make it clearer
n_year_storms_raw = n_year_storms_raw[['n', 'duration_hrs', 'start_time', 'end_time', 'inches']]
n_year_storms_raw.head()

unique_storms = pd.DataFrame(n_year_storms_raw[0:1])
unique_storms.head()

# This method takes in a start time and end time, and searches unique_storms to see if a storm with these times
# overlaps with anything already in unique_storms.  Returns True if it overlaps with an existing storm
def overlaps(start_time, end_time):
    Range = namedtuple('Range', ['start', 'end'])
    range_to_check = Range(start=start_time, end=end_time)
    for index, row in unique_storms.iterrows():
        date_range = Range(start=row['start_time'], end=row['end_time'])
        latest_start = max(range_to_check.start, date_range.start)
        earliest_end = min(range_to_check.end, date_range.end)
        if  ((earliest_end - latest_start).days + 1) > 0:
            return True
    return False
s = pd.to_datetime('1987-08-11 01:00:00')
e = pd.to_datetime('1987-08-11 23:59:00')
overlaps(s,e)

# Iterate through n_year_storms_raw and if an overlapping storm does not exist in unique_storms, then add it
for index, storm in n_year_storms_raw.iterrows():
    if not overlaps(storm['start_time'], storm['end_time']):
        unique_storms = unique_storms.append(storm)
unique_storms.head()

# How many of each n-year storm did we see?
unique_storms['n'].value_counts().sort_index()

unique_storms.dtypes

def find_year(timestamp):
    return timestamp.year
unique_storms['year'] = unique_storms['start_time'].apply(find_year)
unique_storms.head()

unique_storms['year'].value_counts().sort_index().plot(kind='bar', title='N-Year Storms per Year')

ns_by_year = {year: {n: 0 for n in n_s} for year in range(1970, 2017)}
for index, event in unique_storms.iterrows():
    ns_by_year[event['year']][int(event['n'])] += 1
ns_by_year = pd.DataFrame(ns_by_year).transpose()
ns_by_year.head()

ns_by_year = ns_by_year[ns_by_year.columns[::-1]]  # Reverse column order
ns_by_year.columns = [str(n) + '-year' for n in ns_by_year.columns]
ns_by_year.plot(kind='bar', stacked=True, title="N-Year Storms by Year")

unique_storms.to_csv('data/n_year_storms_ohare_noaa.csv', index=False)



