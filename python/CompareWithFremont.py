import os

# Download Pronto Data
if not os.path.exists('open_data_year_one.zip'):
    get_ipython().system('curl -O https://s3.amazonaws.com/pronto-data/open_data_year_one.zip')
    get_ipython().system('unzip open_data_year_one.zip')

# Download Spokane St. Bridge data
if not os.path.exists('SpokaneBridge.csv'):
    get_ipython().system('curl -o SpokaneBridge.csv https://data.seattle.gov/api/views/upms-nr8w/rows.csv?accessType=DOWNLOAD')

# Download Fremont Bridge data
if not os.path.exists('FremontBridge.csv'):
    get_ipython().system('curl -o FremontBridge.csv https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()

FB_trips = pd.read_csv('FremontBridge.csv', index_col='Date',
                       parse_dates=['Date'])
FB_by_date = FB_trips.groupby(FB_trips.index.date.astype('datetime64')).sum()
FB_by_date['Fremont Bridge Total'] = FB_by_date.sum(1)
FB_by_date.head()

SS_trips = pd.read_csv('SpokaneBridge.csv', index_col='Date',
                       parse_dates=['Date'])
SS_by_date = SS_trips.groupby(SS_trips.index.date.astype('datetime64')).sum()
SS_by_date = SS_by_date.iloc[:, [1, 2, 0]]
SS_by_date.columns = ['Spokane St. Bridge West',
                      'Spokane St. Bridge East',
                      'Spokane St. Bridge Total']
SS_by_date.head()

fig, ax = plt.subplots(2, figsize=(16, 6), sharex=True);
FB_by_date['2015'].iloc[:, :2].plot(ax=ax[0]);
SS_by_date['2015'].iloc[:, :2].plot(ax=ax[1]);

PCS_trips = pd.read_csv('2015_trip_data.csv',
                        parse_dates=['starttime', 'stoptime'],
                        infer_datetime_format=True)

# Find the start date
ind = pd.DatetimeIndex(PCS_trips.starttime)
PCS_by_date = PCS_trips.pivot_table('trip_id', aggfunc='count',
                                    index=ind.date.astype('datetime64'),
                                    columns='usertype')
PCS_by_date.columns.name = None
PCS_by_date['Pronto Total'] = PCS_by_date.sum(1)
PCS_by_date.head()

joined = PCS_by_date.join(FB_by_date).join(SS_by_date)
joined.head()

fig, ax = plt.subplots(figsize=(16, 6))
joined[['Pronto Total', 'Spokane St. Bridge Total', 'Fremont Bridge Total']].plot(ax=ax);

fig, ax = plt.subplots(figsize=(16, 6), sharex=True, sharey=True)
cols = ['Annual Member', 'Short-Term Pass Holder']

for i, col in enumerate(cols):
    ratio = (joined[col] / joined['Fremont Bridge Total'])
    ratio.plot(ax=ax, lw=1, alpha=0.4)
    smoothed = pd.rolling_window(ratio, 14, win_type='gaussian', std=3, center=True, min_periods=7)
    smoothed.plot(ax=ax, color=ax.lines[-1].get_color(), lw=3)
ax.legend(ax.lines[1::2], cols, loc='upper left')
ax.set_title('Ratio of Daily Pronto Trips to Fremont Bridge Trips')

fig.savefig('figs/compare_with_fremont.png', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(16, 6), sharex=True, sharey=True)
cols = ['Annual Member', 'Short-Term Pass Holder']

for i, col in enumerate(cols):
    ratio = (joined[col] / joined['Spokane St. Bridge Total'])
    ratio.plot(ax=ax, lw=1, alpha=0.4)
    smoothed = pd.rolling_window(ratio, 14, win_type='gaussian', std=3, center=True, min_periods=7)
    smoothed.plot(ax=ax, color=ax.lines[-1].get_color(), lw=3)
plt.legend(ax.lines[1::2], cols, loc='upper left')
plt.title('Ratio of Daily Pronto Trips to Spokane St. Bridge Trips');

