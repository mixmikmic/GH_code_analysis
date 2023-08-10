get_ipython().magic('matplotlib inline')
import matplotlib.pylab
import pandas as pd
import numpy as np

with open('data/ao_monthly.txt') as f:
    for x in range(5):
        print(next(f))

data = pd.read_fwf('data/ao_monthly.txt', header = None)

# Not so great
data.head()

# %load snippets/readtime.py

data = pd.read_fwf('data/ao_monthly.txt', header = None, index_col = 0, parse_dates = [[0,1]], infer_datetime_format= True)

data.head()

data.index

data.index.names = ['Month']
data.columns = ['Value']
data.head()

# %load snippets/daterange.py

# The time span of this dataset is:

min(data.index)
max(data.index)

# data.to_period()
data

# %load snippets/changerep.py
data.to_period()

import timeit
# First, let's see how to use a date_parser:
dateparse = lambda x, y: pd.datetime.strptime('%s-%s'%(x,y), '%Y-%m')

get_ipython().magic("timeit data = pd.read_fwf('data/ao_monthly.txt', header = None, index_col = 0, parse_dates = [[0, 1]], date_parser = dateparse)")

# new in pandas 0.18
df = pd.DataFrame({'year':[2015, 2016], 'month':[2,3], 'day':[4,5], 'hour':[12, 13]})
df

pd.to_datetime(df)

ts = pd.Series(range(10), index = pd.date_range('7/31/15', freq = 'M', periods = 10))
ts

# truncating preserves frequency
ts.truncate(before = '10/31/2015', after = '12/31/2015')

# You can truncate in a way that does not preserve frequency
ts[[1, 6, 7]].index

# But Pandas will try to preserve frequency automatically whenever possible
ts[0:10:2]

