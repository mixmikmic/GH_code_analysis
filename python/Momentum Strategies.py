# Import libraries to find linear trend and plot data
from statsmodels import regression
import statsmodels.api as sm
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import auquanToolbox.dataloader as dl

# Load pricing data for an asset
start = '2014-01-01'
end = '2015-01-01'
data = dl.load_data_nologs('nasdaq', ['AAPL'], start, end)
prices = data['ADJ CLOSE']
dates = prices.index

# Plot the price of the asset over time
plt.figure(figsize=(15,7))
plt.plot(prices['AAPL'])

# Find the line of best fit to illustrate the trend
X = np.arange(len(dates))
x = sm.add_constant(X) # Add a column of ones so that line can have a y-intercept
model = regression.linear_model.OLS(prices['AAPL'], x).fit()
a = model.params[0] # Get coefficients of line
b = model.params[1]
prices['Y_hat'] = X * b + a
plt.plot(prices['Y_hat'], 'r', alpha=0.9);
plt.ylabel('Price')
plt.legend(['AAPL', 'Trendline']);
plt.show()

def generate_autocorrelated_values(N):
    X = np.zeros(N)
    for i in range(N-1):
        X[i+1] = X[i] + np.random.normal(0, 1)
    return X

plt.figure(figsize=(15,7))
for i in range(10):
    X = generate_autocorrelated_values(100)
    plt.plot(X)
plt.xlabel('$t$')
plt.ylabel('$X_t$');
plt.show()

def generate_autocorrelated_values(N):
    X = np.zeros(N)
    for i in range(1, N-1):
        # Do the past returns 'look good' to investors
        past_returns = X[i] - X[i-1]
        # Investors hypothesize that future returns will be equal to past returns and buy at that price
        X[i+1] = X[i] + past_returns + np.random.normal(0, 1)
    return X

plt.figure(figsize=(15,7))
for i in range(10):
    X = generate_autocorrelated_values(10)
    plt.plot(X)
plt.xlabel('$t$')
plt.ylabel('$X_t$');
plt.show()

from statsmodels.tsa.stattools import adfuller

X1 = generate_autocorrelated_values(100)
X2 = np.random.normal(0, 1, 100)

# Compute the p-value of the Dickey-Fuller statistic to test the null hypothesis that yw has a unit root
print 'X1'
_, pvalue, _, _, _, _ = adfuller(X1)
if pvalue > 0.05:
    print 'We cannot reject the null hypothesis that the series has a unit root.'
else:
    print 'We reject the null hypothesis that the series has a unit root.'
print 'X2'
_, pvalue, _, _, _, _ = adfuller(X2)
if pvalue > 0.05:
    print 'We cannot reject the null hypothesis that the series has a unit root.'
else:
    print 'We reject the null hypothesis that the series has a unit root.'

# Load pricing data for an asset
start = '2016-03-01'
end = '2017-01-01'
data = dl.load_data_nologs('nasdaq', ['AAPL'], start, end)
prices = data['ADJ CLOSE']
dates = prices.index

# Plot the price of the asset over time
plt.figure(figsize=(15,7))
plt.plot(prices['AAPL'])

# Find the line of best fit to illustrate the trend
X = np.arange(len(dates))
x = sm.add_constant(X) # Add a column of ones so that line can have a y-intercept
model = regression.linear_model.OLS(prices['AAPL'], x).fit()
a = model.params[0] # Get coefficients of line
b = model.params[1]
prices['Y_hat'] = X * b + a
plt.plot(prices['Y_hat'], 'r', alpha=0.9);
plt.ylabel('Price')
plt.legend(['AAPL', 'Trendline']);
plt.show()

plt.figure(figsize=(15,7))
plt.plot((prices['AAPL'] - prices['Y_hat']).values)
plt.hlines(np.mean(prices['AAPL'] - prices['Y_hat']), 0, len(dates), colors='r')
plt.hlines(np.std(prices['AAPL'] - prices['Y_hat']), 0, len(dates), colors='r', linestyles='dashed')
plt.hlines(-np.std(prices['AAPL'] - prices['Y_hat']), 0, len(dates), colors='r', linestyles='dashed')
plt.xlabel('Time')
plt.ylabel('Dollar Difference')
plt.show()

