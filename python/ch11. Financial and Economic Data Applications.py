from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
pd.options.display.max_rows = 12
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 4))

get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

close_px = pd.read_csv('data/ch11/stock_px.csv', parse_dates=True, index_col=0)
volume = pd.read_csv('data/ch11/volume.csv', parse_dates=True, index_col=0)
prices = close_px.loc['2011-09-05':'2011-09-14', ['AAPL', 'JNJ', 'SPX', 'XOM']]
volume = volume.loc['2011-09-05':'2011-09-12', ['AAPL', 'JNJ', 'XOM']]

# 거래량
volume
prices * volume

# 단일 거래에 대한, 거래된 한주당 평균 주가 
vwap = (prices * volume).sum() / volume.sum()
vwap

vwap.dropna()

prices.align(volume, join='inner')

s1 = Series(range(3), index=['a', 'b', 'c'])
s2 = Series(range(4), index=['d', 'b', 'c', 'e'])
s3 = Series(range(3), index=['f', 'a', 'c'])
DataFrame({'one': s1, 'two': s2, 'three': s3})

DataFrame({'one': s1, 'two': s2, 'three': s3}, index=list('face'))

ts1 = Series(np.random.randn(3),
             index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))
ts1

ts1.resample('B')

ts1.resample('B').ffill()

dates = pd.DatetimeIndex(['2012-6-12', '2012-6-17', '2012-6-18',
                          '2012-6-21', '2012-6-22', '2012-6-29'])
ts2 = Series(np.random.randn(6), index=dates)
ts2

# ts2로 다시 인덱싱 하겠다. 
ts1.reindex(ts2.index).ffill()

ts2 + ts1.reindex(ts2.index).ffill()

gdp = Series([1.78, 1.94, 2.08, 2.01, 2.15, 2.31, 2.46],
             index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
infl = Series([0.025, 0.045, 0.037, 0.04],
              index=pd.period_range('1982', periods=4, freq='A-DEC'))
gdp
infl

infl_q = infl.asfreq('Q-SEP', how='end')
infl_q

infl_q.reindex(gdp.index, method='ffill')
infl_q



