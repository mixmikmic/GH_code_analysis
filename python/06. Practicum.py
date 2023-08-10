get_ipython().magic('matplotlib inline')
import matplotlib.pylab
import numpy as np
import pandas as pd

from pandas_datareader import data, wb
start = pd.Timestamp('2016-1-1')
end = pd.Timestamp('2017-1-1')
f = data.DataReader("F", 'yahoo', start, end)

from pandas_datareader import data, wb
start = pd.Timestamp('2016-1-1')
end = pd.Timestamp('2017-1-1')
f = data.DataReader("F", 'yahoo', start, end)

f.head()

# %load snippets/prac1.py
f['2016-07':'2016-08'][['High', 'Low']].plot()

# %load snippets/prac2.py
r = f.rolling(50).var()['Volume'].plot()

# %load snippets/prac3.py
r = f.expanding().var()['Volume'].plot()

# %load snippets/prac4.py
lagged = f.shift(1)
sum((f - lagged)['Open'] > 0)
f['DayGain'] = f['Open'] - lagged['Open']
sum(f['DayGain'] > 0)/len(f['DayGain'])

# %load snippets/prac5.py
f.rolling(window = 25)['DayGain'].apply(lambda x: len([x_i for x_i in x if x_i > 0])/len(x)).plot()

# %load snippets/prac6.py
f.resample('M').mean()['High'].plot()

# %load snippets/prac7.py
volume = f.Volume
volume_lagged = f.Volume.shift()
diffed_volume = volume - volume_lagged
diffed_volume.rolling(window = 20).var().plot()

# %load snippets/prac8.py
# What's the best predictor of tomorrow's stock price?
pd.DataFrame({'real':f.Volume, 'lagged':f.Volume.shift()}).corr()



