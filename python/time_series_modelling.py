import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('../dataset/train.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.head()

# TODO

# TODO

# TODO

# TODO

# TODO

# TODO

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

# TODO

# TODO

# TODO

# TODO

# TODO

# TODO

