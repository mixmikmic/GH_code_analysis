from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
get_ipython().magic('matplotlib inline')

# Get precipitation since 2000
rain_df = pd.read_csv('../n-year/notebooks/data/ohare_full_precip_hourly.csv')
rain_df['datetime'] = rain_df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
rain_df = rain_df.set_index(pd.DatetimeIndex(rain_df['datetime']))
rain_df = rain_df['2000-01-01':][['HOURLYPrecip', 'datetime']].copy()
rain_df.head()

daily_rain = rain_df['HOURLYPrecip'].resample('1D').sum()
daily_rain = pd.DataFrame(daily_rain)
daily_rain.head()

wib_df = pd.read_csv('311_data/wib_calls_311_comm.csv')
wib_df.head()

wib_df = pd.read_csv('311_data/wib_calls_311_comm.csv')
wib_df['WIB_Calls'] = wib_df[wib_df.columns.values[1:]].sum(axis=1)
wib_sum = wib_df[['Created Date', 'WIB_Calls']].copy()
wib_sum['Created Date'] = pd.to_datetime(wib_sum['Created Date'])
wib_sum = wib_sum.set_index(wib_sum['Created Date'])
wib_sum = wib_sum[['WIB_Calls']]
wib_sum.head()

rain_wib = daily_rain.copy()
rain_wib['WIB_Calls'] = wib_sum['WIB_Calls']
rain_wib.head()

plt.rcParams["figure.figsize"] = [15, 5]
fix, ax = plt.subplots()
rain_plot = rain_wib['HOURLYPrecip'].plot(ax=ax, style='blue', label='Daily Precipitation')
wib_plot = rain_wib['WIB_Calls'].plot(ax=ax, style='red', secondary_y=True, label='WIB Calls')

rain_patch = mpatches.Patch(color='blue', label='Daily Precipitation')
wib_patch = mpatches.Patch(color='red', label='Basement Flooding Calls')
plt.legend(handles=[rain_patch, wib_patch])
plt.title("Precipitation and Basement Flooding Over Time")
plt.show()



