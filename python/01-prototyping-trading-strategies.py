get_ipython().magic('matplotlib inline')

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# set defauly plotting size
import matplotlib.pylab
matplotlib.pylab.rcParams['figure.figsize'] = (12, 7)

# helper function for candlestick plotting
import matplotlib.dates as mdates
try:
    from mpl_finance import candlestick_ohlc
except ImportError:
    from matplotlib.finance import candlestick_ohlc

def plot_candlestick(df, ax=None, fmt="%Y-%m-%d", cols=["open", "high", "low", "close"]):
    if ax is None:
        fig, ax = plt.subplots()
        
    idx_name = df.index.name
    dat = df.reset_index()[[idx_name]+cols]
    dat[df.index.name] = dat[df.index.name].map(mdates.date2num)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.xticks(rotation=45)
    _ = candlestick_ohlc(ax, dat.values, width=.6, colorup='g', alpha =1)
    ax.set_xlabel(idx_name)
    ax.set_ylabel("Price")
    return ax

from pandas_datareader import data as pdr
import fix_yahoo_finance

spy = pdr.get_data_yahoo("SPY", start="2000-01-01")
spy.head()

ratio = spy["Close"] / spy["Adj Close"]

spy["close"]  = spy["Adj Close"]
spy["open"]   = spy["Open"] / ratio
spy["high"]   = spy["High"] / ratio
spy["low"]    = spy["Low"] / ratio
spy["volume"] = spy["Volume"]

spy = spy[['open','high','low','close','volume']]
spy.head()

# saving data
spy.to_csv('~/Desktop/sp500_ohlc.csv')

# how many rows/columns do we have?
spy.shape

# calculate returns
spy['return'] = spy['close'].pct_change()
spy['return'].describe()

# slicing
spy[5:10][["close", "return"]]

# slicing using queries
spy[ spy['return'] > 0.005 ]['return'].describe()

# simple plotting
spy['close'].plot()

# histogram plot
spy['return'].plot.hist(bins=100, edgecolor='white')

# resampling to 1 year (A=Annual)
spy['return'].resample("1A").sum() * 100

# using pandas' rolling mean
spy['ma1'] = spy['close'].rolling(window=50).mean()
spy['ma2'] = spy['close'].rolling(window=200).mean()

spy[['ma1', 'ma2', 'close']].plot(linewidth=1)

# rolling standard deviation
spy['return'].rolling(window=20).std().plot()

# calculating log returns and volatility
spy['logret'] = np.log(spy['close'] / spy['close'].shift(1))
spy['volatility'] = spy['logret'].rolling(window=252).std() * np.sqrt(252)

spy[['close', 'volatility']].plot(subplots=True)

# pure python bollinger bands
spy['sma'] = spy['close'].rolling(window=20).mean()
spy['std'] = spy['close'].rolling(window=20).std()

spy['upperbb'] = spy['sma'] + (spy['std'] * 2)
spy['lowerbb'] = spy['sma'] - (spy['std'] * 2)

ax = plot_candlestick(spy[-100:])
ax.plot(spy[-100:][['upperbb', 'sma', 'lowerbb']], linewidth=1)

# using TA-Lib..
import talib as ta

spy['rsi'] = ta.RSI(spy['close'].values, timeperiod=2)
spy[-100:][['close', 'rsi']].plot(subplots=True)

# having some fun with plotting :)

data = spy[-100:]

fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])

ax0 = plt.subplot(gs[0])
plt.plot(data['close'])
ax0.set_ylabel('close')
ax0.fill_between(data.index, data['close'].min(), data['close'], alpha=.25)

ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.set_ylabel('RSI 2')
ax1.plot(data['rsi'], color="navy", linewidth=1)
ax1.axhline(90, color='r', linewidth=1.5)
ax1.axhline(10, color='g', linewidth=1.5)

portfolio = pd.DataFrame(data={ 'spy': spy['return'] })
portfolio['strategy'] = portfolio[ portfolio['spy'].shift(1) <= -0.005 ]['spy']

# plot strategy
portfolio.fillna(0).cumsum().plot()

# Annualized Sharpe Ratio
def sharpe(returns, periods=252, riskfree=0):
    returns = returns.dropna()
    return np.sqrt(periods) * (np.mean(returns-riskfree)) / np.std(returns)

# benchmark sharpe
sharpe(portfolio['spy'])

# strategy sharpe
sharpe(portfolio['strategy'])

# time in market
len(portfolio['strategy'].dropna()) / len(portfolio)

# EOY Returns
eoy = portfolio.resample("A").sum()
eoy['diff'] = eoy['strategy']/eoy['spy']

print( np.round(eoy[['spy', 'strategy', 'diff']] * 100, 2) )

ma_portfolio = spy[['close', 'return']].copy()
ma_portfolio.rename(columns={'return':'spy'}, inplace=True)

# create the moving averages
ma_portfolio['ma1'] = ma_portfolio['close'].rolling(window=50).mean()
ma_portfolio['ma2'] = ma_portfolio['close'].rolling(window=200).mean()

# strategy rules
ma_portfolio['position'] = np.where(ma_portfolio['ma1'].shift(1) > ma_portfolio['ma2'].shift(1), 1, -1)
ma_portfolio['strategy'] = ma_portfolio['position'] * ma_portfolio['spy']

# plot
ma_portfolio[['strategy', 'spy']].cumsum().plot()

# benchmark sharpe
sharpe(ma_portfolio['spy'])

# strategy sharpe
sharpe(ma_portfolio['strategy'])

# time in market
len(ma_portfolio['strategy'].dropna()) / len(ma_portfolio)

eoy = ma_portfolio.resample("A").sum()
eoy['diff'] = eoy['strategy']/eoy['spy']
print( np.round(eoy[['spy', 'strategy', 'diff']] * 100, 2) )



