from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from datetime import datetime, timedelta
import operator
import matplotlib.pyplot as plt
from collections import namedtuple
get_ipython().magic('matplotlib inline')

# N-Year Storm variables
# These define the thresholds laid out by bulletin 70, and transfer mins and days to hours
n_year_threshes = pd.read_csv('../../n-year/notebooks/data/n_year_definitions.csv')
n_year_threshes = n_year_threshes.set_index('Duration')
dur_str_to_hours = {
    '5-min':0.0833,
    '10-min':0.1667,
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
n_year_threshes = n_year_threshes.iloc[::-1]
n_year_threshes

n_year_threshes.transpose().loc[['1-year', '10-year', '100-year']].plot(kind='bar', title='Rainfall durations for n-year storms')

n_year_threshes[['1-year', '10-year', '100-year']].plot(kind='bar', title='How much rainfaill it takes for n-years given duration')

# Convert durations to hours, so that they make more sense on an axis
n_year_threshes['duration'] = n_year_threshes.index.values
def find_duration_hours(duration_str):
    return dur_str_to_hours[duration_str]
n_year_threshes['hours_duration'] = n_year_threshes['duration'].apply(find_duration_hours)
n_year_threshes = n_year_threshes.drop('duration', 1)
n_year_threshes = n_year_threshes.set_index('hours_duration')
n_year_threshes.head()

n_year_threshes.plot(kind='line', title='Duration vs Inches for the N-Year Storm')

# This method takes in a number of inches, and plots various durations and how they're classified as n-year storms
# as a bar chart
def inches_to_storm(inches):
    ret_val = []
    thresholds = n_year_threshes.transpose()
    for storm in list(thresholds.index.values):
        the_storm = thresholds.loc[storm]
        storms_higher = the_storm.loc[the_storm > inches]
        if len(storms_higher) == 0:
            continue
        upper_hours = the_storm.loc[the_storm >= inches].index[0]
        upper_inches = the_storm.loc[the_storm >= inches].iloc[0]
        try:
            lower_hours = the_storm.loc[the_storm < inches].iloc[::-1].index[0]
            lower_inches = the_storm.loc[the_storm < inches].iloc[::-1].iloc[0]
        except:
            lower_hours = 0
            lower_inches = 0
        percent_across = (inches-lower_inches) / (upper_inches - lower_inches)
        duration = lower_hours + ((upper_hours - lower_hours) * percent_across)
        ret_val.append({'storm': storm, 'hours': duration})
    ret_val = pd.DataFrame(ret_val)
    ret_val = ret_val.set_index('storm')
    ret_val.plot(kind='bar', title='%s Inches over Duration to Classify the N-Year Storms' % str(inches))

inches_to_storm(3.2)



