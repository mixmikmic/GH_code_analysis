import pandas as pd
import quandl

import numpy as np
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk

from scipy import optimize

get_ipython().magic('pylab inline')

n = 2000
data = quandl.get("CHRIS/ICE_B1")[-n:]
returns = data.Change

plt.plot(returns)

model = pm.Model()
with model:
    sigma = pm.Exponential('sigma', 1./.02, testval=.1)

    nu = pm.Exponential('nu', 1./10)
    s = GaussianRandomWalk('s', sigma**-2, shape=n)

    r = pm.StudentT('r', nu, lam=pm.math.exp(-2*s), observed=returns)

with model:
    trace = pm.sample(2000)

figsize(12,6)
pm.traceplot(trace, model.vars[:-1]);

import seaborn as sns

figsize(12,6)
title(str(s))
plot(trace[s][::10].T,'b', alpha=.03);
xlabel('time')
ylabel('log volatility')



plot(np.abs(numpyMatrix))
plot(np.exp(trace[s][::10].T), 'g', alpha=.01);
sd = np.exp(trace[s].T)
axes = plt.gca()
axes.set_xlim([0,2000])
axes.set_ylim([0,6])

xlabel('time')
ylabel('absolute returns')

plt.plot(data.Settle)



