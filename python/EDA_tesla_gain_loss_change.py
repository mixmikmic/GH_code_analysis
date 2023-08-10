import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf

stock = pd.read_csv('TESLAHistoricalQuotes.csv')
stock = stock.drop(0,0)

# Fix up the date column
stock.date = pd.to_datetime(stock.date)
stock = stock.sort_values('date')

stock.set_index(stock.date, inplace=True)
stock = stock.drop('date', 1)

stock['daily_gain'] = stock.close - stock.open

stock['daily_change'] = stock.daily_gain / stock.open

stock.head()

stock.daily_gain.plot(figsize=(12,8), title= 'Tesla Daily Price Change (Dollars)', fontsize=14)
plt.show()

stock.daily_change.plot(figsize=(12,8), title= 'Tesla Daily Price Change (Percent)', fontsize=14)
plt.show()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    # This can be fine tuned.  There is some relationship, but hard to quantify.
    rolmean = pd.rolling_mean(timeseries, window=30)
    rolstd = pd.rolling_std(timeseries, window=30)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in list(dftest[4].items()):
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(stock.daily_gain.dropna(inplace=False))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(stock.daily_gain, lags=366, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.daily_gain, lags=366, ax=ax2)
plt.show()

fig = plt.figure(figsize=(12,8))
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.daily_gain, lags=55, ax=ax2)
plt.show()

test_stationarity(stock.daily_change.dropna(inplace=False))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(stock.daily_change, lags=366, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.daily_change, lags=366, ax=ax2)
plt.show()

stock['target'] = stock.daily_gain.shift(-1)
stock.head()

plt.scatter(stock.volume, stock.target)
plt.show()

plt.scatter(stock.high, stock.target)
plt.show()

plt.scatter(stock.low, stock.target)
plt.show()

plt.scatter(stock.daily_gain, stock.target)
plt.show()

plt.scatter(stock.daily_change, stock.target)
plt.show()



