from statsmodels import regression
import statsmodels.api as sm
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import auquanToolbox.dataloader as dl

# Fetch prices data for 10 stocks from different sectors and plot prices
start = '2014-06-01'
end = '2016-12-31'
assets = ['AAPL', 'AIG', 'C', 'T', 'PG', 'JNJ', 'EOG', 'MET', 'DOW', 'AMGN']
data = dl.load_data_nologs('nasdaq', assets, start, end)
prices = data['ADJ CLOSE']

prices.plot(figsize=(15,7), color=['r', 'g', 'b', 'k', 'c', 'm', 'orange',
                                  'chartreuse', 'slateblue', 'silver'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Price')
plt.xlabel('Time')
plt.show()

asset = prices.iloc[:, 8]
asset.plot(figsize=(15,7))
plt.ylabel('Price')
plt.show()

short_mavg = asset.rolling(window=30, center=False).mean()
long_mavg = asset.rolling(window=200, center=False).mean()

asset[200:].plot(figsize=(15,7))
short_mavg[200:].plot()
long_mavg[200:].plot()
plt.ylabel('Price')
plt.show()

asset.plot(alpha = 1)

rolling_means = {}

for i in np.linspace(10, 100, 10):
    X = asset.rolling(window=int(i),center=False).mean()
    rolling_means[i] = X
    X.plot(figsize=(15,7), alpha = 0.55)
    
rolling_means = pd.DataFrame(rolling_means).dropna()
plt.show()

scores = pd.Series(index=asset.index)
for date in rolling_means.index:
    mavg_values = rolling_means.loc[date]
    ranking = stats.rankdata(mavg_values.values)
    d = distance.hamming(ranking, range(1, 11))
    scores[date] = d
    
# Normalize the  score
(scores).plot(figsize=(15,7), alpha=0.6)
plt.legend(['Signal'], bbox_to_anchor=(1.25, 1))
asset.plot(secondary_y=True, alpha=1)
plt.legend(['Asset Price'])
plt.show()

scores = pd.Series(index=asset.index)
for date in rolling_means.index:
    mavg_values = rolling_means.loc[date]
    ranking = stats.rankdata(mavg_values.values)
    d, _ = stats.spearmanr(ranking, range(1, 11))
    scores[date] = d

# Normalize the  score
(scores).plot(figsize=(15,7), alpha=0.6);
plt.legend(['Signal'], bbox_to_anchor=(1, 0.9))
asset.plot(secondary_y=True, alpha=1)
plt.legend(['Asset Price'], bbox_to_anchor=(1, 1))
plt.show()

scores = pd.Series(index=asset.index)
for date in rolling_means.index:
    mavg_values = rolling_means.loc[date]
    d = np.max(mavg_values) - np.min(mavg_values)
    scores[date] = d
    
# Normalize the  score
(scores).plot(figsize=(15,7), alpha=0.6);
plt.legend(['Signal'], bbox_to_anchor=(1, 0.9))
asset.plot(secondary_y=True, alpha=1)
plt.legend(['Asset Price'], bbox_to_anchor=(1, 1))
plt.show()

k = 30
start = '2014-01-01'
end = '2015-01-01'

x = np.log(asset)
v = x.diff()
m =  data['VOLUME'].iloc[:,8]

p0 = v.rolling(window=k, center=False).sum()
p1 = m*v.rolling(window=k, center=False).sum()
p2 = p1/m.rolling(window=k, center=False).sum()
p3 = v.rolling(window=k, center=False).mean()/v.rolling(window=k, center=False).std()

f, ax1 = plt.subplots(figsize=(15,7))
ax1.plot(p0)
ax2 = ax1.twinx()
ax2.plot(p1,'r')
ax1.plot(p2)
ax1.plot(p3)
ax1.set_title('Momentum of AMGN')
ax1.legend(['p(0)', 'p(2)', 'p(3)'], bbox_to_anchor=(1.25, 1))
ax2.legend(['p(1)'], bbox_to_anchor=(1.25, .75))

plt.show()

