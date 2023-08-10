get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import pandas as pd
df = pd.read_csv('TrackBen.csv', parse_dates=[3])
df.columns=['lat','lng','height','datetime']
df.dtypes

import geo
from geo import circle_dist
df['dist'] = circle_dist(df['lat'], df['lng'], df['lat'].shift(), df['lng'].shift())

df['time_diff'] = df['datetime'] - df['datetime'].shift()

df = df.set_index('datetime')
df['speed'] = df['dist'] / (df['time_diff'] / np.timedelta64(1, 'h'))
df.head()

#resample into 10 minute intervals
mdf = df.resample('10T').mean()

mdf[['height', 'speed']].plot(subplots=True)



