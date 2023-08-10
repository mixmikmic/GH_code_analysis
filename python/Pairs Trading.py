import numpy as np
import pandas as pd

import statsmodels
from statsmodels.tsa.stattools import coint
# just set the seed for the random number generator
np.random.seed(107)

import matplotlib.pyplot as plt

X_returns = np.random.normal(0, 1, 100) # Generate the daily returns
# sum them and shift all the prices up into a reasonable range
X = pd.Series(np.cumsum(X_returns), name='X') + 50
X.plot(figsize=(15,7))
plt.show()

some_noise = np.random.normal(0, 1, 100)
Y = X + 5 + some_noise
Y.name = 'Y'
pd.concat([X, Y], axis=1).plot(figsize=(15,7))
plt.show()

(Y/X).plot(figsize=(15,7)) # Plot the ratio
plt.axhline((Y/X).mean(), color='red', linestyle='--') # Add the mean
plt.xlabel('Time')
plt.legend(['Price Ratio', 'Mean'])
plt.show()

# compute the p-value of the cointegration test
# will inform us as to whether the ratio between the 2 timeseries is stationary
# around its mean
score, pvalue, _ = coint(X,Y)
print pvalue

X.corr(Y)

X_returns = np.random.normal(1, 1, 100)
Y_returns = np.random.normal(2, 1, 100)

X_diverging = pd.Series(np.cumsum(X_returns), name='X')
Y_diverging = pd.Series(np.cumsum(Y_returns), name='Y')

pd.concat([X_diverging, Y_diverging], axis=1).plot(figsize=(15,7))
plt.show()

print 'Correlation: ' + str(X_diverging.corr(Y_diverging))
score, pvalue, _ = coint(X_diverging,Y_diverging)
print 'Cointegration test p-value: ' + str(pvalue)

Y2 = pd.Series(np.random.normal(0, 1, 1000), name='Y2') + 20
Y3 = Y2.copy()

# Y2 = Y2 + 10
Y3[0:100] = 30
Y3[100:200] = 10
Y3[200:300] = 30
Y3[300:400] = 10
Y3[400:500] = 30
Y3[500:600] = 10
Y3[600:700] = 30
Y3[700:800] = 10
Y3[800:900] = 30
Y3[900:1000] = 10

Y2.plot(figsize=(15,7))
Y3.plot()
plt.ylim([0, 40])
plt.show()

# correlation is nearly zero
print 'Correlation: ' + str(Y2.corr(Y3))
score, pvalue, _ = coint(Y2,Y3)
print 'Cointegration test p-value: ' + str(pvalue)

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

import auquanToolbox.dataloader as dl

start = '2013-01-01'
end = '2016-12-31'

m = ['SPX','AAPL','ADBE','SYMC','YHOO','EBAY','MSFT','QCOM','HPQ','JNPR','AMD','IBM']
data = dl.load_data_nologs('nasdaq', m , start, end)
prices = data['ADJ CLOSE']

prices.head()

# Heatmap to show the p-values of the cointegration test between each pair of
# stocks. Only show the value in the upper-diagonal of the heatmap

scores, pvalues, pairs = find_cointegrated_pairs(prices)
import seaborn
seaborn.heatmap(pvalues, xticklabels=m, yticklabels=m, cmap='RdYlGn_r' 
                , mask = (pvalues >= 0.98)
                )
plt.show()
print pairs

S1 = prices['ADBE']
S2 = prices['MSFT']
score, pvalue, _ = coint(S1, S2)
pvalue

ratios = S1 / S2
ratios.plot(figsize=(15,7))
plt.axhline(ratios.mean(), color='black')
plt.legend(['Price Ratio'])
plt.show()

def zscore(series):
    return (series - series.mean()) / np.std(series)

zscore(ratios).plot(figsize=(15,7))
plt.axhline(zscore(ratios).mean(), color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Ratio z-score', 'Mean', '+1', '-1'])
plt.show()

# Get the price ratio between the 2 stocks
ratios = S1 / S2
ratios.name = 'ratio'

# Get the 10 day moving average of the price ratio
ratios_mavg10 = ratios.rolling(window=10,center=False).mean()
ratios_mavg10.name = 'ratio 10d mavg'

# Get the 60 day moving average
ratios_mavg60 = ratios.rolling(window=60,center=False).mean()
ratios_mavg60.name = 'ratio 60d mavg'

plt.figure(figsize=(15,7))
plt.plot(ratios.index, ratios.values)
plt.plot(ratios_mavg10.index, ratios_mavg10.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)

plt.legend(['Ratio','10 Day Ratio MAVG', '60 Day Ratio MAVG'])

plt.ylabel('Ratio')
plt.show()

# Take a rolling 60 day standard deviation
std_60 = ratios.rolling(window=60,center=False).std()
std_60.name = 'std 60d'

# Compute the z score for each day
zscore_60_10 = (ratios_mavg10 - ratios_mavg60)/std_60
zscore_60_10.name = 'z-score'

plt.figure(figsize=(15,7))
zscore_60_10.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()

# Plot the ratios and buy and sell signals from z score
plt.figure(figsize=(15,7))

ratios[60:].plot()
buy = ratios.copy()
sell = ratios.copy()
buy[zscore_60_10>-1] = 0
sell[zscore_60_10<1] = 0
buy[60:].plot(color='g', linestyle='None', marker='^')
sell[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

# Plot the prices and buy and sell signals from z score
plt.figure(figsize=(18,9))
S1[60:].plot(color='b')
S2[60:].plot(color='c')
buyR = 0*S1.copy()
sellR = 0*S1.copy()
# When buying the ratio, buy S1 and sell S2
buyR[buy!=0] = S1[buy!=0]
sellR[buy!=0] = S2[buy!=0]
# When selling the ratio, sell S1 and buy S2 
buyR[sell!=0] = S2[sell!=0]
sellR[sell!=0] = S1[sell!=0]

buyR[60:].plot(color='g', linestyle='None', marker='^')
sellR[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,min(S1.min(),S2.min()),max(S1.max(),S2.max())))

plt.legend(['S1','S2', 'Buy Signal', 'Sell Signal'])
plt.show()

