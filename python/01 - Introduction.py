get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-deep', 'seaborn-darkgrid', 'seaborn-notebook'])
pd.options.display.max_rows = 8

data = pd.read_csv('data/airbase_data.csv', index_col=0, parse_dates=True, na_values=[-9999])

data

data['1999':].resample('A').mean().plot(ylim=[0,100])

exceedances = data > 200
exceedances = exceedances.groupby(exceedances.index.year).sum()
ax = exceedances.loc[2005:].plot(kind='bar')
ax.axhline(18, color='k', linestyle='--')

data['weekday'] = data.index.weekday
data['weekend'] = data['weekday'].isin([5, 6])
data_weekend = data.groupby(['weekend', data.index.hour])['FR04012'].mean().unstack(level=0)
data_weekend.plot()



