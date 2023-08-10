import pandas_datareader.data as wb
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)
BAC = wb.DataReader('BAC', 'yahoo', start, end)
C = wb.DataReader('C', 'yahoo', start, end)
GS = wb.DataReader('GS', 'yahoo', start, end)
JPM = wb.DataReader('JPM', 'yahoo', start, end)
MS = wb.DataReader('MS', 'yahoo', start, end)
WFC = wb.DataReader('WFC', 'yahoo', start, end)

BAC.head()

tickers = 'BAC C GS JPM MS WFC'.split()
tickers

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],  axis=1, keys=tickers)
bank_stocks.head()

bank_stocks.columns.names = ['Bank Ticker','Stock Info']

bank_stocks.head()

bank_stocks.xs('Close', axis=1, level='Stock Info').max()



returns = pd.DataFrame()

for i in tickers:
    returns[i+' Return'] = bank_stocks.xs('Close', level=1, axis=1)[i].pct_change()

returns.head()

sns.pairplot(returns[1:])



returns.idxmin()



returns.idxmax()



returns.std()



#returns.loc['2006-01-03']
#or returns.loc['2015-01-01':'2015-12-31'].std()
returns[returns.index.year == 2015].std()



MS_2015 = returns[returns.index.year == 2015]['MS Return']
sns.distplot(MS_2015, bins=100, color='g')
plt.grid()



C_2008 = returns[returns.index.year == 2008]['C Return']
sns.distplot(C_2008, bins=100, color='r')

plt.grid()
plt.box('on')



import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()

#for i in tickers:
#    bank_stocks[i]['Close'].plot(label=i, figsize=(12,4)).legend()

bank_stocks.xs('Close', level=1, axis=1).plot(figsize=(12,4))





data_2008 = bank_stocks[bank_stocks.index.year == 2008]
close_2008 = data_2008.xs('Close', level=1, axis=1)
rol_30 = close_2008.rolling(window=30).mean()

plt.figure(figsize=(12,6))
plt.plot(close_2008['BAC'], label='BAC Close')
plt.plot(rol_30['BAC'], label='30-day moving avg')
plt.legend()
plt.xlabel('Date')



close = bank_stocks.xs('Close', level=1, axis=1)

sns.heatmap(close.corr(), annot=True, cmap='coolwarm')



sns.clustermap(close.corr(), annot=True, cmap='coolwarm')



