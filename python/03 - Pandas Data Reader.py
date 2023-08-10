import pandas as pd

from pandas_datareader import data

from datetime import datetime

facebook = data.DataReader('FB', 'yahoo', datetime(2007,1, 1), datetime(2016, 1, 1))

facebook.head()

google = data.DataReader('GOOGL', 'yahoo', datetime(2007,1, 1), datetime(2016, 1, 1))

google.head()

google['Stock'] = 'GOOGL'

facebook['Stock'] = 'FB'

pd.concat([google, facebook])

merged = pd.concat([google, facebook])

get_ipython().magic('pylab inline')

merged['Adj Close'].plot()

merged.describe()

merged['High'] > 1200

merged[merged['High'] > 1200]

merged.to_excel('../../data/stocks.xlsx')

get_ipython().magic('load solutions/stocks_solution.py')

