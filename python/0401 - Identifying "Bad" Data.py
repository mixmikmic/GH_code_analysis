import pandas as pd
import numpy as np
from datetime import datetime

weather = pd.read_csv('data/berlin_weather_oldest.csv')

weather.columns

weather.dtypes

weather = weather.set_index('STATION')

weather.head()

weather = weather.applymap(lambda x: np.nan if x == -9999 else x)

weather.head()

weather['DATE'] = weather['DATE'].map(lambda x: datetime.strptime(str(x), '%Y%m%d').date())

weather.notnull().head()

weather.dropna()

weather.dropna(how='all', axis=1)

weather.dropna(thresh=4, axis=1)

weather.shape

weather.dropna(thresh=100, axis=1)

weather['DATE'].dtype

weather = weather.set_index(pd.DatetimeIndex(weather['DATE']))

weather.head()

weather.resample('D').pad()

weather.index.duplicated()

weather[weather.index > datetime(1992, 12, 22)]

weather['STATION_NAME'].value_counts()

weather.index.drop_duplicates().sort_values()

weather.groupby('STATION_NAME').resample('D').mean().head()

weather[weather.PRCP >= weather.PRCP.std() * 3 + weather.PRCP.mean()]

rainy = weather[weather.PRCP >= weather.PRCP.std() * 3 + weather.PRCP.mean()]

rainy['month'] = rainy.index.month

get_ipython().magic('pylab inline')

rainy.groupby('month')['PRCP'].sum().plot()

rainy.max()

rainy[rainy.PRCP > 1000]



