get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']

import pandas as pd
import simplejson as json
from simplejson import JSONDecodeError

with open('../results/run11/driver.json') as fh:
    json_data = [json.loads(row) for row in fh]
df = pd.DataFrame(json_data).rename(columns={'command': 'program'})
df['n'] = df['n'].astype('int')

for task in  ['select', 'filter','groupby', 'load', 'sort', 'join']:
    group = df.groupby(['n', 'program'])[task]
    m = group.mean().unstack()
    e = group.std().unstack()
    ax = m.plot(kind='bar', logy=True, title=task, alpha=.75)
    ax.set_ylabel("seconds")

sqlite_load = pd.Series({
    1000: 0.024,
    10000: 0.068,
    100000: 0.499,
    1000000: 4.740,
    10000000: 46.121,
})

pandas_load = df.groupby(['program', 'n'])['load'].mean().loc['pandas']

pd.DataFrame({
    'sqlite': sqlite_load,
    'pandas': pandas_load
}).plot(kind='bar', alpha=.75, logy=True)

def graph(df, labels, n):
    group = df.groupby(['n', 'program'])[labels]
    m = group.mean().loc[n].T
    s = group.std().loc[n].T
    ax = m.plot(kind='bar', yerr=s, alpha=0.75)
    ax.set_ylabel("seconds")
    print m
    
graph(df, ['groupby', 'load'], 1000000)
graph(df, ['select', 'filter'], 1000000)

