import warnings
warnings.simplefilter('ignore')

get_ipython().magic('matplotlib inline')

import time
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from __future__ import division

mpl.style.use('ggplot')
figsize = (14,7)

today = time.strftime("%Y-%m-%d")
print('This notebook was last rendered on {}.'.format(today))

hits_df = pd.read_csv('ipynb_counts.csv', index_col=0, header=0, parse_dates=True)
hits_df.reset_index(inplace=True)
hits_df.drop_duplicates(subset='date', inplace=True)
hits_df.set_index('date', inplace=True)
hits_df.sort_index(ascending=True, inplace=True)

hits_df.tail(3)

til_today = pd.date_range(hits_df.index[0], hits_df.index[-1])

hits_df = hits_df.reindex(til_today)

ax = hits_df.plot(title="GitHub search hits for {} days".format(len(hits_df)), figsize=figsize)
ax.set_xlabel('Date')
ax.set_ylabel('# of ipynb files')

daily_deltas = (hits_df.hits - hits_df.hits.shift()).fillna(0)

outliers = abs(daily_deltas - daily_deltas.mean()) > 2.5*daily_deltas.std()

hits_df.ix[outliers] = np.NaN

hits_df = hits_df.interpolate(method='time')

ax = hits_df.plot(title="GitHub search hits for {} days sans outliers".format(len(hits_df)), 
                  figsize=figsize)
ax.set_xlabel('Date')
_ = ax.set_ylabel('# of ipynb files')

total_delta_nbs = hits_df.iloc[-1] - hits_df.iloc[0]
total_delta_nbs

avg_delta_nbs = total_delta_nbs / len(hits_df)
avg_delta_nbs

daily_deltas = (hits_df.hits - hits_df.hits.shift()).fillna(0)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(pd.rolling_mean(daily_deltas, window=30, min_periods=0), 
        label='30-day rolling mean of daily-change')
ax.plot(daily_deltas, label='24-hour change')
ax.set_xlabel('Date')
ax.set_ylabel('Delta notebook count')
ax.set_title('Change in notebook count')
_ = ax.legend(loc='upper left')

fig, ax = plt.subplots(figsize=figsize)
ax.plot(pd.rolling_mean(daily_deltas, window=30, min_periods=0))
ax.set_xlabel('Date')
ax.set_ylabel('Delta notebook count')
_ = ax.set_title('30-day rolling mean of daily-change')

def train(df):
    ar_model = sm.tsa.AR(df, freq='D')
    ar_model_res = ar_model.fit(ic='bic')
    return ar_model_res

start_date='2014-10-20'
end_date='2018-01-01'
model_dates = [today, '2017-01-01', '2016-01-01']

models = [train(hits_df.loc[:date]) for date in model_dates]

pd.DataFrame([m.params for m in models], index=model_dates).T

predictions = [model.predict(start=start_date, end=end_date, dynamic=True) for model in models]

eval_df = pd.DataFrame(predictions, index=model_dates).T

fig, ax = plt.subplots(figsize=figsize)
ax.set_title('GitHub search hits predicted from {} until {}'.format(start_date, end_date))
# plot the raw search numbers
ax.plot(hits_df, 'ko', markersize=1, label='truth')
# use the pandas plotting api mostly because it formats the legend for us
ax.plot(eval_df, linewidth=2)
# call to ax.legend so that the 'truth' label shows up
ax.legend(['truth'] + list(eval_df.columns))
_ = ax.set_ylabel('# of ipynb files')

eval_df['truth'] = hits_df.hits
residual_df = -eval_df.subtract(eval_df.truth, axis=0).dropna().drop('truth', axis=1)
_ = eval_df.drop('truth', axis=1)

fig, ax = plt.subplots(figsize=figsize)
ret = ax.plot(residual_df, 'o', ms=2)
ax.legend(residual_df.columns)
ax.set_ylabel('# of ipynb files')
ax.set_title('Residuals between predicted and truth')
fig.autofmt_xdate()

