import pandas as pd

stock = pd.read_csv('TESLAHistoricalQuotes.csv')
stock = stock.drop(0,0)

stock.head()

stock.tail()

stock.dtypes

# Fix up the date column
stock.date = pd.to_datetime(stock.date)
stock = stock.sort_values('date')
stock.set_index(stock.date, inplace=True)
stock = stock.drop('date', 1)

stock.head()

from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.ticker as mtick

stock.close.plot(figsize=(12,8), title= 'Tesla Closing Price', fontsize=14)
plt.show()

decomposition = seasonal_decompose(stock.close, freq=366)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
plt.savefig('foo.png', bbox_inches='tight')
plt.show()

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    # Super not sure how to determin this... We'll have to do some research
    # 7 to 14 days seems reasonable to find overall trends
    rolmean = pd.rolling_mean(timeseries, window=14)
    rolstd = pd.rolling_std(timeseries, window=14)

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

test_stationarity(stock.close)

stock['first_difference'] = stock.close - stock.close.shift(1)  
test_stationarity(stock.first_difference.dropna(inplace=False))

stock['second_difference'] = stock.close - stock.close.shift(2)  
test_stationarity(stock.second_difference.dropna(inplace=False))

stock['seasonal_difference'] = stock.close - stock.close.shift(366)
test_stationarity(stock.seasonal_difference.dropna(inplace=False))

stock['seasonal_first_difference'] = stock.first_difference - stock.first_difference.shift(366)  
test_stationarity(stock.seasonal_first_difference.dropna(inplace=False))

import statsmodels.api as sm 
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(stock.close.iloc[366:], lags=366, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.close.iloc[366:], lags=366, ax=ax2)
plt.show()

fig = plt.figure(figsize=(12,8))
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(stock.close.iloc[366:], lags=80, ax=ax2)
plt.show()

from pandas.tools.plotting import autocorrelation_plot

autocorrelation_plot(stock.close)
plt.show()

stock['other_seasonal_difference'] = stock.close - stock.close.shift(80)
test_stationarity(stock.other_seasonal_difference.dropna(inplace=False))

decomposition = seasonal_decompose(stock.close, freq=80)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
plt.show()

fig, ax = plt.subplots()
ax=stock[['close']].iloc[1640:].plot(figsize=(12,8), title= 'Tesla Closing Price', fontsize=14, ax=ax)
plt.title('Tesla Closing Price', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Stock Price', fontsize=16)
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 
plt.yticks()
plt.savefig('foo.png', bbox_inches='tight')
plt.show()

decomposition = seasonal_decompose(stock.close, freq=16)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
plt.show()



def rolmean_plot(timeseries):
    
    #Determing rolling statistics
    # Super not sure how to determin this... We'll have to do some research
    # 7 to 14 days seems reasonable to find overall trends
    rolmean3 = pd.rolling_mean(timeseries, window=3)
    rolmean7 = pd.rolling_mean(timeseries, window=7)
    rolmean14 = pd.rolling_mean(timeseries, window=14)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[1740:], color='blue',label='Original')
    mean3 = plt.plot(rolmean3.iloc[1740:], color='red', label='Rolling Mean 3 Days')
    mean7 = plt.plot(rolmean7.iloc[1740:], color='green', label='Rolling Mean 7 Days')
    mean14 = plt.plot(rolmean14.iloc[1740:], color='orange', label='Rolling Mean 14 Days')
    plt.legend(loc='best')
    plt.title('Rolling Mean Comparisons',fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.savefig('rolmean.png', bbox_inches='tight')
    plt.show()

rolmean_plot(stock.close)



