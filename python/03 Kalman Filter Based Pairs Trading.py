get_ipython().run_line_magic('matplotlib', 'inline')
# Imports
from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Jupyter")
AddReference("QuantConnect.Indicators")
from System import *
from QuantConnect import *
from QuantConnect.Data.Market import TradeBar, QuoteBar
from QuantConnect.Jupyter import *
from QuantConnect.Indicators import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
# Create an instance
qb = QuantBook()
# plt.style.available

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import statsmodels.api as sm
plt.style.use('seaborn-whitegrid')

symbols = ["XOM","CVX"]
data = {}
plt.figure(figsize =(15,7))
for syl in symbols:
    qb.AddEquity(syl)
    data[syl] = qb.History(syl, datetime(2003,1,1), datetime(2009,1,1), Resolution.Daily).loc[syl]['close']
    data[syl].index = data[syl].index.date
    data[syl].plot(label=syl)
plt.legend()
plt.ylabel('Adjusted Price')
plt.figure(figsize =(15,7))
for syl in symbols:    
    (1+data[syl].pct_change()).cumprod().plot(label=syl)
plt.legend()
plt.ylabel('Cumulative Return')

# Run linear regression over two log price series
x = list(np.log(data[symbols[0]]))
x_const = sm.add_constant(x)
y = list(np.log(data[symbols[1]]))
linear_reg = sm.OLS(y,x_const)
results = linear_reg.fit()
results.summary()

beta = results.params[1]
alpha = results.params[0]

y_pred = np.log(data[symbols[0]])*beta + alpha

plt.figure(figsize =(15,7))
cm = plt.get_cmap('jet')
sc = plt.scatter(np.log(data[symbols[0]]), np.log(data[symbols[1]]), s=50, c=x, cmap=cm, marker='o',
                 alpha=0.6,label='Log Price',edgecolor='k')
plt.plot(x, y_pred, '-',c='black',linewidth=3, label='OLS Fit')
plt.legend()
cb = plt.colorbar(sc)
cb.ax.set_yticklabels([str(p) for p in data[symbols[0]].index])
plt.xlabel(symbols[0])
plt.ylabel(symbols[1])

# construct the spread series according to OLS result
df = pd.DataFrame(np.log(data[symbols[1]]) - np.log(data[symbols[0]])*beta-alpha,index=data[symbols[0]].index)
df.columns = ['spread']
df.plot(figsize =(15,10))
plt.ylabel('spread')

# check if the spread is stationary 
adf = sm.tsa.stattools.adfuller(df['spread'], maxlag=1)
print 'ADF test statistic: %.02f' % adf[0]
for key, value in adf[4].items():
    print('\t%s: %.3f' % (key, value))
print 'p-value: %.03f' % adf[1]

df['mean'] = df['spread'].mean()
df['upper'] = df['mean'] + 1.96*df['spread'].std()
df['lower'] = df['mean'] - 1.96*df['spread'].std()

df.plot(figsize =(15,10),style=['g', '--r', '--b', '--b'])

df['buy'] = df['spread'][((df['spread'] < df['lower']) & (df['spread'].shift(1) > df['lower']) | 
                          (df['spread'] <  df['mean']) & (df['spread'].shift(1) >  df['mean']))]

df['sell'] = df['spread'][((df['spread'] > df['upper']) & (df['spread'].shift(1) < df['upper']) | 
                           (df['spread'] >  df['mean']) & (df['spread'].shift(1) <  df['mean']))]
df.plot(figsize =(17,10), style=['g', '--r', '--b', '--b', 'm^','cv'])

obs_mat = sm.add_constant(np.log(data[symbols[0]]).values, prepend=False)[:, np.newaxis]
trans_cov = 1e-5 / (1 - 1e-5) * np.eye(2)

kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                  initial_state_mean=np.ones(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=0.5,
                  transition_covariance=0.000001 * np.eye(2))
# np.array([[ 0.5,  0.5 ],[0.5 ,  0.5]])

state_means, state_covs = kf.filter(np.log(data[symbols[1]]).values)
slope=state_means[:, 0] 
intercept=state_means[:, 1]
plt.figure(figsize =(15,7))
plt.plot(data[symbols[1]].index, slope, c='b')
plt.ylabel('slope')
plt.figure(figsize =(15,7))
plt.plot(data[symbols[1]].index,intercept,c='r')
plt.ylabel('intercept')

# visualize the correlation between assest prices over time
plt.figure(figsize =(15,7))
cm = plt.get_cmap('jet')
sc = plt.scatter(np.log(data[symbols[0]]), np.log(data[symbols[1]]), s=50, c=x, cmap=cm, marker='o',
                 alpha=0.6,label='Price',edgecolor='k')
cb = plt.colorbar(sc)
cb.ax.set_yticklabels([str(p) for p in data[symbols[0]].index])
plt.xlabel("price (%s)" %symbols[0])
plt.ylabel("price (%s)" %symbols[0])

# add regression lines
step = 50 # pick slope and intercept every 50 days
colors_l = np.linspace(0.1, 1, len(state_means[::step]))
for i, b in enumerate(state_means[::step]):
    plt.plot(np.log(data[symbols[0]]), b[0] *np.log(data[symbols[0]]) + b[1], alpha=.5, lw=2, c=cm(colors_l[i]))

kl_spread = np.log(data[symbols[1]]) - np.log(data[symbols[0]]) * state_means[:,0] - state_means[:,1]
df['kl_spread'] = kl_spread

new_df = df.drop(['buy','sell'],1)
new_df.plot(figsize =(15,8),style=['r','--y', '--y', '--y','b'])



