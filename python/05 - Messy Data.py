import pandas as pd
import numpy as np
from datetime import datetime

weather = pd.read_csv('../../data/berlin_weather_oldest.csv')

weather.dtypes

weather.head()

weather = weather.applymap(lambda x: np.nan if x == -9999 else x)

weather.head()

weather['DATE'] = weather['DATE'].map(lambda x: datetime.strptime(str(x), '%Y%m%d').date())

weather['DATE']

weather.notnull().head()

weather.dropna()

weather.dropna(how='all', axis=1)

weather.shape

weather.dropna(thresh=weather.shape[0] * .1, axis=1)

weather = weather.set_index(pd.DatetimeIndex(weather['DATE']))

weather.head()

weather.index.duplicated()

weather['STATION_NAME'].value_counts()

weather.index.drop_duplicates().sort_values()

weather.groupby('STATION_NAME').resample('D').mean().head()

rainy = weather[weather.PRCP >= weather.PRCP.std() * 3 + weather.PRCP.mean()]

rainy['month'] = rainy.index.month

get_ipython().magic('pylab inline')

rainy.groupby('month')['PRCP'].sum().plot()

get_ipython().magic('load solutions/weather_solution_rainyday.py')



get_ipython().magic('load solutions/weather_solution_fix_stations.py')



