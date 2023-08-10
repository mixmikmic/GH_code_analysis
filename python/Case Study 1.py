import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.style.use('seaborn-whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')

from pandas_datareader import data

returns = data.get_data_yahoo('SPY',  start='2008-05-01', end='2009-12-01')['Close'].pct_change()

len(returns)

returns.head()

returns.plot(figsize=(10, 6))
plt.ylabel('daily returns in %')



with pm.Model() as sp500_model:
    nu = pm.Exponential('nu', 1./10, testval=5.)
    sigma = pm.Exponential('sigma', 1./.02, testval=.1)

    s = pm.GaussianRandomWalk('s', sigma**-2, shape=len(returns))
    volatility_process = pm.Deterministic('volatility_process', pm.math.exp(-2*s))

    r = pm.StudentT('r', nu, lam=1/volatility_process, observed=returns)



with sp500_model:
    trace = pm.sample(2000)

## We can check our samples by looking at the traceplot for nu and sigma

pm.traceplot(trace, [nu, sigma])

fig, ax = plt.subplots(figsize=(15, 8))
returns.plot(ax=ax)
ax.plot(returns.index, 1/np.exp(trace['s',::5].T), 'r', alpha=.03);
ax.set(title='volatility_process', xlabel='time', ylabel='volatility');
ax.legend(['S&P500', 'stochastic volatility process'])













