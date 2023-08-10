URL = 'https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD'

from urllib.request import urlretrieve
urlretrieve(URL, 'Fremont.csv')

import pandas as pd
data = pd.read_csv('Fremont.csv', index_col='Date', parse_dates=True)
data.head()

get_ipython().magic('matplotlib inline')
data.resample('W').sum().plot();



