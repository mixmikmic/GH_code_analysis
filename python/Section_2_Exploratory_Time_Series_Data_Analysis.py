get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

import warnings; warnings.simplefilter('ignore')
from __future__ import absolute_import, division, print_function

import sys
import os

import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.5f' % x)

np.set_printoptions(precision=5,suppress=True)

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

sns.set(style='ticks',context='poster')

Sentiment = 'data/sentiment.csv'
Sentiment = pd.read_csv(Sentiment,index_col=0,parse_dates=[0])

C = 'data/citi.csv'
C = pd.read_csv(C, index_col=0, parse_dates=[0])

T10yr = 'data/T10yr.csv'
T10yr = pd.read_csv(T10yr, index_col=0, parse_dates=[0])

bentley = pd.read_csv('data/bentley_bbq_03.csv', index_col=0, parse_dates=[0])

print("Citigroup's stock price:", "\n", C.dtypes, "\n")
print("10 Year Treasury Bond Rate:", "\n", T10yr.dtypes, "\n")
print("University of Michigan: Consumer Sentiment:", "\n", Sentiment.dtypes)

C.close = C['Close']
T10yr.close = T10yr['Close']

bentley.head()

Sentiment.head()

# Select the series from 2005 - 2016
sentiment_short = Sentiment.ix['2005':'2016']

sentiment_short.index[:5]

print(sentiment_short.dtypes)

sentiment_short.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title('Consumer Sentiment')
sns.despine()

import datetime as dt
parser = lambda date: pd.datetime.strptime(date, '%d/%m/%Y')
bentley_data = pd.read_csv('data/bentley_bbq_03.csv',index_col=0, date_parser=parser)

bentley_data.head()

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sentiment_short, lags=20, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sentiment_short, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout();

lags = 9
ncols = 3
nrows = int(np.ceil(lags/ncols))

fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows)) 

for ax, lag in zip(axes.flat, np.arange(1, lags+1, 1)):
    
    lag_str = 't-{}'.format(lag)
    X = (pd.concat([sentiment_short, sentiment_short.shift(-lag)], axis=1,
                   keys=['y'] + [lag_str]).dropna())

    X.plot(ax=ax, kind='scatter', y='y', x=lag_str);
    corr = X.corr().as_matrix()[0][1]
    ax.set_ylabel('Original')
    ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
    ax.set_aspect('equal');
    sns.despine();
    
fig.tight_layout()

def tsplot(y, lags=None, title='', figsize=(14,8)):
    fig = plt.figure(figsize=figsize)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout, (0,0))
    hist_ax = plt.subplot2grid(layout, (0,1))
    acf_ax = plt.subplot2grid(layout, (1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y,lags=lags,ax=acf_ax)
    smt.graphics.plot_pacf(y,lags=lags,ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

tsplot(sentiment_short, title='Consumer Sentiment', lags=36)

tsplot(bentley_data, title='Consumer Sentiment', lags=36)



