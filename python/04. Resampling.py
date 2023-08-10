import pandas as pd
import numpy as np

rng = pd.date_range('1/1/2011', periods = 72, freq = 'H')
ts = pd.Series(list(range(len(rng))), index = rng)

converted = ts.asfreq('45Min', method = 'ffill')
converted

ts.shape

converted.shape

ts.asfreq('45 min')

converted[1:10]

ts[1:10]



converted = ts.asfreq('3H')

converted[1:10]

ts[1:10]

# Let's try the more flexible .resample()
ts.resample('2H').mean()[1:10]

# What's particularly useful is that we can use reample to event out irregular time series
irreg_ts = ts[list(np.random.choice(a = list(range(len(ts))), size = 10, replace = False))]

irreg_ts

irreg_ts.asfreq('D')

irreg_ts = irreg_ts.sort_index()
irreg_ts

irreg_ts.asfreq('D')

irreg_ts.resample('D').count()

irreg_ts.fillna(method = 'ffill',limit = 5)

