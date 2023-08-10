get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from jupyterworkflow.data import get_fremont_data

data = get_fremont_data()
data.head()

data.resample('W').sum().plot();

data.groupby(data.index.time).mean().plot();

pivoted = data.pivot_table('Total', index=data.index.time, columns=data.index.date)
pivoted.iloc[:5, :5]

pivoted.plot(legend=False, alpha=0.01);



