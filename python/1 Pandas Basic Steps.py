import pandas as pd
import numpy as np
from datetime import datetime

x = pd.DataFrame({'Min': [3.2, 2.1, 7.4, 5.2, 3.9, 0.7], 
                  'Max': [5.3, 3.7, 11.2, 8.1, 9.2, 3.2], 
                  'Avg': [4.7, 3.0, 8.1, 7.2, 5.1, 1.8]},
                    index=[datetime(2017,8,1), datetime(2017,8,2), datetime(2017,8,3), 
                           datetime(2017,8,4), datetime(2017,8,5), datetime(2017,8,6)])

x

# type information
x.dtypes

# information
x.info()

# accessing columns
x.Max

x['Avg']

# accessing rows
x.loc['2017-08-01']

x.iloc[0]

x['2017-08-02':'2017-08-03']

x[['Avg', 'Min']]

x[1:3]

get_ipython().magic('matplotlib inline')

x.plot()

# Introducing missing data points into the data
x = pd.DataFrame({'Min': [3.2, 2.1, None, 5.2, 3.9, 0.7], 
                  'Max': [5.3, None, 11.2, 8.1, 9.2, 3.2], 
                  'Avg': [4.7, 3.0, 8.1, 7.2, 5.1, 1.8]},
                    index=[datetime(2017,8,1), datetime(2017,8,2), datetime(2017,8,3), 
                           datetime(2017,8,4), datetime(2017,8,5), datetime(2017,8,6)])

x

# removing rows with missing values
x.dropna() # alternatively, use axis=0

# remove columns which have missing data
x.dropna(axis=1)

x.fillna(0.0)

x.fillna(method='pad')

x.fillna(method='bfill')

# resample to a specific time
# we don't have a lot of data, but let's use two days
x.asfreq('12H')

x.asfreq('2D')

# resample returns a resampling object that has an aggregate method
x.resample('2D')

x.resample('2D').agg(np.min)

x.resample('2D').asfreq()

x.resample('12H').ffill()



