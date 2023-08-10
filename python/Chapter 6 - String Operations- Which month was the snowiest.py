get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.mpl_style', 'default')
plt.rcParams['figure.figsize'] = (15, 3)
plt.rcParams['font.family'] = 'sans-serif'

weather_2012 = pd.read_csv('../data/weather_2012.csv', parse_dates=True, index_col='Date/Time')
weather_2012[:5]

weather_description = weather_2012['Weather']
is_snowing = weather_description.str.contains('Snow')

# Not super useful
is_snowing[:5]

# More useful!
is_snowing.plot()

weather_2012['Temp (C)'].resample('M').apply(np.median).plot(kind='bar')

is_snowing.astype(float)[:10]

is_snowing.astype(float).resample('M').apply(np.mean)

is_snowing.astype(float).resample('M').apply(np.mean).plot(kind='bar')

temperature = weather_2012['Temp (C)'].resample('M').apply(np.median)
is_snowing = weather_2012['Weather'].str.contains('Snow')
snowiness = is_snowing.astype(float).resample('M').apply(np.mean)

# Name the columns
temperature.name = "Temperature"
snowiness.name = "Snowiness"

stats = pd.concat([temperature, snowiness], axis=1)
stats

stats.plot(kind='bar')

stats.plot(kind='bar', subplots=True, figsize=(15, 10))

