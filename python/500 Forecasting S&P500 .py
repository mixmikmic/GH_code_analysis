get_ipython().magic('pylab inline')
import numpy as np
import pandas as pd

data = pd.read_csv('./data/GSPC.csv', index_col=0, na_values=0)
data.index = pd.to_datetime(data.index)
data.head()

data.plot(y=['Open', 'High', 'Low', 'Close', 'Adj Close'], figsize=(14, 4))

