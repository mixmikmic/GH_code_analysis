import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Load data
# Note that since dates and hours are *truncated*, not rounded, we want to translate each data point to 
# the midpoint of the day or hour. Hence, adding a Timedelta.
df1 = pd.read_csv("/mnt/data/shared/aws-data/restricted-data/CDR-data/cust_foreigners_timeseries_GEN.csv")
df1['date_'] = pd.to_datetime(df1['date_'], format='%Y-%m-%d %H:%M:%S') + pd.Timedelta(hours=12)
df2 = pd.read_csv("/mnt/data/shared/aws-data/restricted-data/CDR-data/cust_foreigners_timeseries_hourly_GEN.csv")
df2['hour_'] = pd.to_datetime(df2['hour_'], format='%Y-%m-%d %H:%M:%S') + pd.Timedelta(minutes=30)

df1.head()

df2.head()

df1['I_calls'] = df1['calls']>=1
df1['I_in_florence'] = df1['in_florence']>=1
df1['I_in_florence_comune'] = df1['in_florence_comune']>=1
df1.head()

df2['I_calls'] = df2['calls']>=1
df2['I_in_florence'] = df2['in_florence']>=1
df2['I_in_florence_comune'] = df2['in_florence_comune']>=1
df2.head()

ts1 = df1.groupby('date_').sum()
ts1.index.name = 'date'
ts1.reset_index(inplace=True)
ts1['date'] = pd.to_datetime(ts1['date'], format='%Y-%m-%d %H:%M:%S')
ts1.head()

ts2 = df2.groupby('hour_').sum()
ts2.index.name = 'hour'
ts2.reset_index(inplace=True)
ts2['hour'] = pd.to_datetime(ts2['hour'], format='%Y-%m-%d')
ts2.head()

highlight_start = ts1['date'][ts1['date'].dt.dayofweek==5]
highlight_end = ts1['date'][ts1['date'].dt.dayofweek==0]

plt.figure(figsize=(20, 10))
ax = plt.gca()

ts2.plot.line(x='hour', y='I_calls', ax=ax, color='black', style='.-')
ts2.plot.line(x='hour', y='I_in_florence', ax=ax, color='red', style='.-')
ts2.plot.line(x='hour', y='I_in_florence_comune', ax=ax, color='blue', style='.-')

# ts1.plot.line(x='date', y='I_calls', ax=ax, color='black', style='.-')
# ts1.plot.line(x='date', y='I_in_florence', ax=ax, color='red', style='.-')
# ts1.plot.line(x='date', y='I_in_florence_comune', ax=ax, color='blue', style='.-')


for i in range(len(highlight_start)):
        ax.axvspan(highlight_start.iloc[i],highlight_end.iloc[i],alpha=.1,color="gray")

plt.axvline('2016-07-01',color="gray")
plt.axvline('2016-08-01',color="gray")
plt.axvline('2016-09-01',color="gray")

ax.legend(labels=['All','In Florence province','In Florence city'])
plt.title('Hourly numbers of unique foreign customers making or receiving calls in CDR data')

ax.set_xlim(['2016-06-01','2016-07-01'])

plt.show()

plt.figure(figsize=(10, 10))
ax = plt.gca()

ts2.plot.line(x='hour', y='I_calls', ax=ax, color='black', style='.-')
ts2.plot.line(x='hour', y='I_in_florence', ax=ax, color='red', style='.-')
ts2.plot.line(x='hour', y='I_in_florence_comune', ax=ax, color='blue', style='.-')

# ts1.plot.line(x='date', y='I_calls', ax=ax, color='black', style='.-')
# ts1.plot.line(x='date', y='I_in_florence', ax=ax, color='red', style='.-')
# ts1.plot.line(x='date', y='I_in_florence_comune', ax=ax, color='blue', style='.-')


for i in range(len(highlight_start)):
        ax.axvspan(highlight_start.iloc[i],highlight_end.iloc[i],alpha=.1,color="gray")

plt.axvline('2016-07-01',color="gray")
plt.axvline('2016-08-01',color="gray")
plt.axvline('2016-09-01',color="gray")

ax.legend(labels=['All','In Florence province','In Florence city'])
plt.title('One week: Hourly numbers of unique foreign customers making or receiving calls in CDR data')

ax.set_xlim(['2016-07-07 12:00:00','2016-07-14 12:00:00'])

plt.show()

plt.figure(figsize=(20, 10))
ax = plt.gca()

ts1.plot.line(x='date', y='I_calls', ax=ax, color='black', style='.-')
ts1.plot.line(x='date', y='I_in_florence', ax=ax, color='red', style='.-')
ts1.plot.line(x='date', y='I_in_florence_comune', ax=ax, color='blue', style='.-')

for i in range(len(highlight_start)):
        ax.axvspan(highlight_start.iloc[i],highlight_end.iloc[i],alpha=.1,color="gray")

plt.axvline('2016-07-01',color="gray")
plt.axvline('2016-08-01',color="gray")
plt.axvline('2016-09-01',color="gray")

ax.legend(labels=['All','In Florence province','In Florence city'])
plt.title('Daily numbers of unique foreign customers making or receiving calls in CDR data')

# ax.set_xlim(['2016-06-01','2016-07-01'])

plt.show()

