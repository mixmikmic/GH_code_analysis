import collections
import json

import cufflinks as cf
import pandas as pd
import plotly.plotly as py
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)
cf.go_offline()

get_ipython().system('sbt "run-main ContinuousDoubleAuction"')

with open('output.json') as simulation_output:    
    data = json.load(simulation_output)
    
df = pd.io.json.json_normalize(data)

df.head()

df.describe()

df['spread'] = df['bidOrder.limit.value'] - df['askOrder.limit.value']

df[['askOrder.limit.value', 'bidOrder.limit.value', 'price.value']].iplot();

df.spread.iplot(logy=True);

returns = df['price.value'].pct_change()
returns.abs().iplot(logy=True);

# limit price for ask orders was sampled from U[1, 2147483647]
df['askOrder.limit.value'].iplot(kind='hist')

