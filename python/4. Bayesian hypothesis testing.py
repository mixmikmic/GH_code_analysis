get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
figsize(16, 9)

melb_temp = pd.read_csv('../Data/melbourne-temp.csv')

melb_temp[:5]

melb_temp['Decimal date'] = melb_temp['Year'] + (melb_temp['Month'] - 1) / 12

t = melb_temp[['Decimal date', 'Mean maximum temperature (°C)']].set_index('Decimal date')

get_ipython().magic('matplotlib inline')

t.plot()

t.loc[1985:1986]

import pymc3 as pm
import numpy as np

with pm.Model() as model2:
    sigma = pm.Uniform('sigma', 0, 15)
    
    a = pm.Uniform('a', -0.1, 0.1)
    b = pm.Uniform('b', 10, 35)
    c = pm.Uniform('c', 10, 15)   # half difference between summer and winter temperature: 10 and 35, say
    d = pm.Uniform('d', 0, 0.2)   # peak summer is somewhere between Jan 01 and end of Feb

    times = np.array(t.index)
    y = pm.Normal('y', a * times + b + c * pm.math.sin(2 * np.pi * (times + d)), sigma,
                  observed=t.values)

with model:
    trace = pm.sample(1000, tune=500)

pm.traceplot(trace);

a_samples = trace['a']
b_samples = trace['b']
c_samples = trace['c']
d_samples = trace['d']
sigma_samples = trace['sigma']

plt.hist(a_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $a$", color="#7A68A6", normed=True);



