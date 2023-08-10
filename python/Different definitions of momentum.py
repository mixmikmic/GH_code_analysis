import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

k = 30
start = '2014-01-01'
end = '2015-01-01'
pricing = get_pricing('PEP', fields='price', start_date=start, end_date=end)
fundamentals = init_fundamentals()
num_shares = get_fundamentals(query(fundamentals.earnings_report.basic_average_shares,)
                              .filter(fundamentals.company_reference.primary_symbol == 'PEP',), end)
x = np.log(pricing)
v = x.diff()
m = get_pricing('PEP', fields='volume', start_date=start, end_date=end)/num_shares.values[0,0]

p0 = pd.rolling_sum(v, k)
p1 = pd.rolling_sum(m*v, k)
p2 = p1/pd.rolling_sum(m, k)
p3 = pd.rolling_mean(v, k)/pd.rolling_std(v, k)

f, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(p0)
ax1.plot(p1)
ax1.plot(p2)
ax1.plot(p3)
ax1.set_title('Momentum of PEP')
ax1.legend(['p(0)', 'p(1)', 'p(2)', 'p(3)'], bbox_to_anchor=(1.1, 1))

ax2.plot(p0)
ax2.plot(p1)
ax2.plot(p2)
ax2.plot(p3)
ax2.axis([0, 300, -0.005, 0.005])
ax2.set_xlabel('Time');

def get_p(prices, m, d, k):
    """ Returns the dth-degree rolling momentum of data using lookback window length k """
    x = np.log(prices)
    v = x.diff()
    m = np.array(m)
    
    if d == 0:
        return pd.rolling_sum(v, k)
    elif d == 1:
        return pd.rolling_sum(m*v, k)
    elif d == 2:
        return pd.rolling_sum(m*v, k)/pd.rolling_sum(m, k)
    elif d == 3:
        return pd.rolling_mean(v, k)/pd.rolling_std(v, k)

# Load the assets we want to trade
start = '2010-01-01'
end = '2015-01-01'
assets = sorted(['STX', 'WDC', 'CBI', 'JEC', 'VMC', 'PG', 'AAPL', 'PEP', 'AON', 'DAL'])
data = get_pricing(assets, start_date='2010-01-01', end_date='2015-01-01').loc['price', :, :]

# Get turnover rate for the assets
fundamentals = init_fundamentals()
num_shares = get_fundamentals(query(fundamentals.earnings_report.basic_average_shares,)
                              .filter(fundamentals.company_reference.primary_symbol.in_(assets),), end)
turnover = get_pricing(assets, fields='volume', start_date=start, end_date=end)/num_shares.values[0]

# Plot the prices just for fun
data.plot(figsize=(10,7), colors=['r', 'g', 'b', 'k', 'c', 'm', 'orange',
                                  'chartreuse', 'slateblue', 'silver'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Price')
plt.xlabel('Time');

# Calculate all the rolling momenta for the data and compute daily ranking of assets by momentum
lookback = 30
ps = np.array([np.array(get_p(data, turnover, j, lookback).T) for j in range(4)])
orders = [[ps[j].T[i].argsort() for i in range(len(ps[0,0]))] for j in range(4)]
ranks = [[orders[j][i].argsort() for i in range(len(orders[1]))] for j in range(4)]

# Cast data to numpy array for easier manipulation
data_array = np.array(data)

# Simulate going long on high-momentum stocks and short low-momentum stocks
# Our first 2*lookback - 2 values will be NaN since we used 2 lookback windows, so start on day 2*lookback
tots = [[0]*4 for j in range(len(data) - 2*lookback)]
for t in range(2*lookback, len(ranks[0]) - 2*lookback):
    tots[t] = list(tots[t-1])
    # Only update portfolio every 2*lookback days
    if t%(2*lookback):
        continue
    # Go long top quintile of stocks and short bottom quintile
    shorts = np.array([[int(x < 2)for x in ranks[j][t]] for j in range(4)])
    longs = np.array([[int(x > 7) for x in ranks[j][t]] for j in range(4)])
    # How many shares of each stock are in $1000
    shares_in_1k = 1000/data_array[t]
    # Go long and short $1000 each in the specified stocks, then clear holdings in 2*lookback days
    returns = (data_array[t+2*lookback]*shares_in_1k - [1000]*len(assets))*(longs - shorts)
    tots[t] += np.sum(returns, 1)

# Adjust so that tots[t] is actually money on day t
tots = [[0,0,0,0]]*2*lookback + tots

# Plot total money earned using the 3 different momentum definitions
plt.plot(tots)
plt.title('Cash in portfolio')
plt.legend(['p(0)', 'p(1)', 'p(2)', 'p(3)'], loc=4)
plt.xlabel('Time')
plt.ylabel('$');

