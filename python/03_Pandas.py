import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

names = ['Ryan', 'Chiara', 'Johnny']
values = [35, 36, 1.8]
ages = pd.Series(values, index=names)
ages

ages.plot(kind='bar')

np.log(ages) / ages**2

ages.index

ages.loc['Johnny']

ages.iloc[2]

ages.values

ages.index

# first we create a dictionary
data = {'age': [35, 36, 1.8],
        'height': [180, 155, 83],
        'weight': [72.5, np.nan, 11.3]}
df = pd.DataFrame(data, index=['Ryan', 'Chiara', 'Johnny'])
df

df.info()

df.min()

df.mean()

df.std()

df.describe()

df['height']

df.height

df['density'] = df.weight / df.height
df

education = pd.Series(['PhD', 'PhD', None, 'masters'],
                     index=['Ryan', 'Chiara', 'Johnny', 'Takaya'],
                     name='education')
# returns a new DataFrame
df.join(education)

# returns a new DataFrame
df.join(education, how='right')

# returns a new DataFrame
df.reindex(['Ryan', 'Chiara', 'Johnny', 'Takaya', 'Kerry'])

adults = df[df.age > 18]
adults

df['is_adult'] = df.age > 18
df

df.plot(kind='scatter', x='age', y='height', grid=True)

df.plot(kind='bar')







two_years = pd.date_range(start='2014-01-01', end='2016-01-01', freq='D')
timeseries = pd.Series(np.sin(2 *np.pi *two_years.dayofyear / 365),
                       index=two_years)
timeseries.plot()

timeseries.loc['2015-01-01':'2015-07-01'].plot()

#!curl -L -o goog.csv http://tinyurl.com/rces-goog
#!curl -L -o aapl.csv http://tinyurl.com/rces-aapl-csv
get_ipython().system(' cp /home/pangeo/notebooks/GOOGL.csv goog.csv')
get_ipython().system(' cp /home/pangeo/notebooks/AAPL.csv aapl.csv')

get_ipython().system(' head goog.csv')

goog = pd.read_csv('goog.csv')
goog.head()

goog = pd.read_csv('goog.csv', parse_dates=[0], index_col=0)
goog.head()

goog.info()

goog.Close.plot()

aapl = pd.read_csv('aapl.csv', parse_dates=[0], index_col=0)
aapl.info()

aapl_close = aapl.Close.rename('aapl')
goog_close = goog.Close.rename('goog')
stocks = pd.concat([aapl_close, goog_close], axis=1)
stocks.head()

stocks.plot()

stocks.corr()

# resample by taking the mean over each month

fig, ax = plt.subplots()
stocks.resample('MS').mean().plot(ax=ax, colors=['r', 'b'])
# and each year
stocks.resample('AS').mean().plot(ax=ax, colors=['r', 'b'])

# get resample object
rs = stocks.goog.resample('MS')
# standard deviation of each month
rs.std().plot()

get_ipython().system(' curl -o nyc_temp.txt http://berkeleyearth.lbl.gov/auto/Local/TAVG/Text/40.99N-74.56W-TAVG-Trend.txt')

get_ipython().system(' head -72 nyc_temp.txt | tail -8')



##### http://berkeleyearth.lbl.gov/locations/40.99N-74.56W
# http://berkeleyearth.lbl.gov/auto/Local/TAVG/Text/40.99N-74.56W-TAVG-Trend.txt


#temp = pd.read_csv('nyc_temp.txt')

col_names = ['year', 'month', 'monthly_anom'] + 10*[]
temp = pd.read_csv('nyc_temp.txt',
                   header=None, usecols=[0, 1, 2], names=col_names,
                   delim_whitespace=True, comment='%')

temp.head()

# need a day
date_df = temp.drop('monthly_anom', axis=1)
date_df['day'] = 1
date_index = pd.DatetimeIndex(pd.to_datetime(date_df))
temp = temp.set_index(date_index).drop(['year', 'month'], axis=1)
temp.head()

temp.plot()

fig, ax = plt.subplots()
temp.plot(ax=ax)
temp.resample('AS').mean().plot(ax=ax)
temp.resample('10AS').mean().plot(ax=ax)

# more advanced operation on rolling windows
def difference_max_min(data):
    return data.max() - data.min()

rw = temp.rolling('365D')
rw.apply(difference_max_min).plot()

# diurnal cycle has been removed!
temp.groupby(temp.index.month).mean().plot()

# find the hottest years
temp.groupby(temp.index.year).mean().sort_values('monthly_anom', ascending=False).head(10)

# https://data.cityofnewyork.us/Health/Rats/amyk-xiv9
rats = pd.read_csv('https://data.cityofnewyork.us/api/views/amyk-xiv9/rows.csv',
                  parse_dates=['APPROVED_DATE', 'INSPECTION_DATE'])

rats.info()

rats.head()

rats.groupby('INSPECTION_TYPE')['INSPECTION_TYPE'].count()

rats.groupby('BORO_CODE')['BORO_CODE'].count().head()

rats.groupby('STREET_NAME')['STREET_NAME'].count().head(20)

# clean up street name
street_names_cleaned = rats.STREET_NAME.str.strip()
street_names_cleaned.groupby(street_names_cleaned).count().head(20)

count = street_names_cleaned.groupby(street_names_cleaned).count()
count.sort_values(ascending=False).head(20)

rats[['LATITUDE', 'LONGITUDE']].describe()

valid_latlon = rats[(rats.LATITUDE > 30) & (rats.LONGITUDE < -70)]
valid_latlon.plot.hexbin('LONGITUDE', 'LATITUDE', C='BORO_CODE', cmap='Set1')

# https://github.com/pandas-dev/pandas/issues/10678
valid_latlon.plot.hexbin('LONGITUDE', 'LATITUDE', sharex=False)

valid_latlon.plot.hexbin('LONGITUDE', 'LATITUDE', sharex=False, bins='log', cmap='magma')

manhattan_rats = valid_latlon[valid_latlon.BORO_CODE==1]
manhattan_rats.plot.hexbin('LONGITUDE', 'LATITUDE', sharex=False, bins='log', cmap='magma')

inspection_date = pd.DatetimeIndex(rats.INSPECTION_DATE)

fig, ax = plt.subplots()
rats.groupby(inspection_date.weekday)['JOB_ID'].count().plot(kind='bar', ax=ax)
ax.set_xlabel('weekday');

fig, ax = plt.subplots()
rats.groupby(inspection_date.hour)['JOB_ID'].count().plot(kind='bar', ax=ax)
ax.set_xlabel('hour');

fig, ax = plt.subplots()
rats.groupby(inspection_date.month)['JOB_ID'].count().plot(kind='bar', ax=ax)
ax.set_xlabel('month')



