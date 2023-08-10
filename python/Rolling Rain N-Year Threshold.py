from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# Loading in hourly rain data from CSV, parsing the timestamp, and adding it as an index so it's more useful

rain_df = pd.read_csv('data/ohare_full_precip_hourly.csv')
rain_df['DATE'] = pd.to_datetime(rain_df['DATE'])
rain_df = rain_df.set_index(pd.DatetimeIndex(rain_df['DATE']))
print(rain_df.dtypes)
rain_df.head()

chi_rain_sub = rain_df[:'1955-01-01']
chi_rain_sub['HOURLYPrecip'].max()

chi_rain_sub = rain_df[rain_df['HOURLYPrecip'] > 0.0]
chi_rain_sub['HOURLYPrecip'].head()

rain_df = rain_df['1970-09-01':]
rain_df.head()

# Resampling the dataframe into one hour increments, accessing max because accumulation listed more often than hourly (i.e. 
# every 15 minutes) is the total precipitation since the hour began
# Description: http://www1.ncdc.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf

chi_rain_series = rain_df['HOURLYPrecip'].resample('1H').max()
print(chi_rain_series.count())
chi_rain_series.head()

roll_6_hr = chi_rain_series.rolling(window=6)
roll_6_hr.sum().plot()

roll_6 = pd.DataFrame(roll_6_hr.sum())

print('For 6-hour intervals')
print('{} 1-year events for Northeast Illinois'.format(len(roll_6[(roll_6['HOURLYPrecip'] >= 1.88) & 
                                                                  (roll_6['HOURLYPrecip'] < 2.28)])))
print('{} 2-year events for Northeast Illinois'.format(len(roll_6[(roll_6['HOURLYPrecip'] >= 2.28) &
                                                                  (roll_6['HOURLYPrecip'] < 2.85)])))
print('{} 5-year events for Northeast Illinois'.format(len(roll_6[(roll_6['HOURLYPrecip'] >= 2.85) &
                                                                  (roll_6['HOURLYPrecip'] < 3.35)])))
print('{} 10-year events for Northeast Illinois'.format(len(roll_6[(roll_6['HOURLYPrecip'] >= 3.35) &
                                                                   (roll_6['HOURLYPrecip'] < 4.13)])))
print('{} 25-year events for Northeast Illinois'.format(len(roll_6[(roll_6['HOURLYPrecip'] >= 4.13) & 
                                                                   (roll_6['HOURLYPrecip'] < 4.90)])))
print('{} 50-year events for Northeast Illinois'.format(len(roll_6[(roll_6['HOURLYPrecip'] >= 4.90) &
                                                                   (roll_6['HOURLYPrecip'] < 5.69)])))
print('{} 100-year events for Northeast Illinois'.format(len(roll_6[roll_6['HOURLYPrecip'] >= 5.69])))

roll_6_1yr = roll_6[(roll_6['HOURLYPrecip'] >= 1.88) & (roll_6['HOURLYPrecip'] < 2.28)]
print('{} days with 1-year events in Northeast Illinois'.format(len(roll_6_1yr.groupby(roll_6_1yr.index.date))))
roll_6_1yr.sort_values(by=['HOURLYPrecip'], ascending=False, inplace=True)
roll_6_1yr.head()

# Many of these are from the same days, but over slightly different intervals as mentioned before
roll_6_2yr = roll_6[(roll_6['HOURLYPrecip'] >= 2.28) & (roll_6['HOURLYPrecip'] < 2.85)]
print('{} days with 2-year events in Northeast Illinois'.format(len(roll_6_2yr.groupby(roll_6_2yr.index.date))))
roll_6_2yr.sort_values(by=['HOURLYPrecip'], ascending=False, inplace=True)
roll_6_2yr.head()

# Helper function taking the series, window, and list of cutoffs to make this quicker, returns the subset
def rolling_results(rain_series, window, rain_cutoffs):
    window_df = pd.DataFrame(rain_series.rolling(window=window).sum())
    print('For {}-hour intervals'.format(window))
    print('{} 1-year events for Northeast Illinois'.format(len(window_df[(window_df['HOURLYPrecip'] >= rain_cutoffs[0]) & 
                                                                         (window_df['HOURLYPrecip'] < rain_cutoffs[1])])))
    print('{} 2-year events for Northeast Illinois'.format(len(window_df[(window_df['HOURLYPrecip'] >= rain_cutoffs[1]) & 
                                                                         (window_df['HOURLYPrecip'] < rain_cutoffs[2])])))
    print('{} 5-year events for Northeast Illinois'.format(len(window_df[(window_df['HOURLYPrecip'] >= rain_cutoffs[2]) & 
                                                                         (window_df['HOURLYPrecip'] < rain_cutoffs[3])])))
    print('{} 10-year events for Northeast Illinois'.format(len(window_df[(window_df['HOURLYPrecip'] >= rain_cutoffs[3]) & 
                                                                         (window_df['HOURLYPrecip'] < rain_cutoffs[4])])))
    print('{} 25-year events for Northeast Illinois'.format(len(window_df[(window_df['HOURLYPrecip'] >= rain_cutoffs[4]) & 
                                                                          (window_df['HOURLYPrecip'] < rain_cutoffs[5])])))
    print('{} 50-year events for Northeast Illinois'.format(len(window_df[(window_df['HOURLYPrecip'] >= rain_cutoffs[5]) & 
                                                                          (window_df['HOURLYPrecip'] < rain_cutoffs[6])])))
    print('{} 100-year events for Northeast Illinois'.format(len(window_df[window_df['HOURLYPrecip'] >= rain_cutoffs[6]])))
    
# Gets the subset of the dataframe for the given cutoff index (i.e. 5 year is the third, so cutoff_index would be 3)
def rolling_subset(rain_series, window, rain_cutoffs, cutoff_index):
    window_df = pd.DataFrame(rain_series.rolling(window=window).sum())
    if cutoff_index <= 6:
        return window_df[(window_df['HOURLYPrecip'] >= rain_cutoffs[cutoff_index - 1]) & (roll_6['HOURLYPrecip'] < 2.85)]
    if cutoff_index == 7:
        return window_df[window_df['HOURLYPrecip'] >= rain_cutoffs[cutoff_index -1]]

cutoffs_12hr = [2.18, 2.64, 3.31, 3.89, 4.79, 5.6, 6.59]
rolling_results(chi_rain_series, 12, cutoffs_12hr)

roll_12_2yr = rolling_subset(chi_rain_series, 12, cutoffs_12hr, 2)
print('{} days with 2-year events for 12 hrs in Northeast Illinois'.format(len(roll_12_2yr.groupby(roll_12_2yr.index.date))))

cutoffs_24hr = [2.51, 3.04, 3.80, 4.47, 5.51, 6.46, 7.58]
rolling_results(chi_rain_series, 24, cutoffs_24hr)

roll_24_1yr = rolling_subset(chi_rain_series, 24, cutoffs_24hr, 1)
print('{} days with 1-year events for 24 hrs in Northeast Illinois'.format(len(roll_24_1yr.groupby(roll_24_1yr.index.date))))

cutoffs_48hr = [2.70, 3.30, 4.09, 4.81, 5.88, 6.84, 8.16]
rolling_results(chi_rain_series, 48, cutoffs_48hr)

roll_48_1yr = rolling_subset(chi_rain_series, 48, cutoffs_48hr, 1)
print('{} days with 1-year events for 48 hrs in Northeast Illinois'.format(len(roll_48_1yr.groupby(roll_48_1yr.index.date))))



