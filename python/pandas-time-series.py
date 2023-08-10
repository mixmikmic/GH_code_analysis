get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
mpl.rcParams['figure.figsize'] = (12.5, 6.0)
import pandas as pd
pd.options.display.max_rows = 12

# parsed with python's dateutil
pd.to_datetime('January, 2017'), pd.to_datetime('3rd of February 2016'), pd.to_datetime('6:31PM, Nov 11th, 2017')

date = pd.to_datetime('3rd of January 2018')
date.strftime('%A')

date + pd.to_timedelta(np.arange(12), 'D')

dates = pd.to_datetime(['3rd of January 2016', '2016-Jul-6', '20170708'])
dates

dates.to_period('D')

dates - dates[0]

pd.date_range('2017-11-29', '2017-12-03')

pd.date_range('2017-12-03', periods=6)

pd.date_range('2017-12-03', periods=10, freq='H')

pd.period_range('2017-09', periods=8, freq='M')

pd.timedelta_range(0, periods=10, freq='H')

data = pd.read_csv('../data/fremont-bridge.csv', index_col='Date', parse_dates=True)
data.head()

data.columns = ['West', 'East']
data['Total'] = data['West'] + data['East']
data.head()

data.info()

data.describe()

data.plot(alpha=0.6)
plt.ylabel('bicycle count');

# probably a better granularity for the full datatset
weekly = data.resample('W').sum()
weekly.plot(style=[':', '--', '-'])
plt.ylabel('bicycle count');

# perhaps a rolling window can be better
daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=[':', '--', '-'])
plt.ylabel('mean daily count');

# and if we do it on week
weekly = data.resample('W').sum()
weekly.rolling(10, center=True).sum().plot(style=[':', '--', '-'])
plt.ylabel('mean weekly count');

series = data.index.to_series()
series

series.dt.dayofyear

series.dt.dayofweek

series.groupby(series.dt.dayofweek).count()

series.groupby(series.dt.year).count()

series2012 = series['2012']
series2012.groupby(series2012.dt.dayofweek).count()

data[(data['West'].isnull()) | (data['East'].isnull())]

# hourly traffic
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks=hourly_ticks, style=[':', '--', '-']);

# weekly traffic
by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
by_weekday.plot(style=[':', '--', '-']);

# is it different on the weekend?
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()
by_time

fig, ax = plt.subplots(2, 1, figsize=(14, 9))
fig.subplots_adjust(hspace=0.3)
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays',
                            xticks=hourly_ticks, style=[':', '--', '-'])
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends',
                            xticks=hourly_ticks, style=[':', '--', '-']);

