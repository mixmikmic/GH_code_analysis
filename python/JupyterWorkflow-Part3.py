URL = 'https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD'

from urllib.request import urlretrieve
urlretrieve(URL, 'Fremont.csv')

import pandas as pd
data = pd.read_csv('Fremont.csv', index_col='Date', parse_dates=True)
data.head()

get_ipython().magic('matplotlib inline')
data.resample('W').sum().plot();

import matplotlib.pyplot as plt
plt.style.use('seaborn')

data.columns = ['West', 'East']

data.resample('W').sum().plot();

data['Total'] = data['West'] + data['East']

ax = data.resample('D').sum().rolling(365).sum().plot();
ax.set_ylim(0, None);

data.groupby(data.index.time).mean().plot();

pivoted = data.pivot_table('Total', index=data.index.time, columns=data.index.date)
pivoted.iloc[:5, :5]

pivoted.plot(legend=False, alpha=0.01);



