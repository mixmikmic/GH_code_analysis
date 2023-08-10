# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get price data
bitcoin = quandl.get("BCHAIN/MKPRU")
sp500 = pd.read_csv('^GSPC.csv', index_col = 0, parse_dates = [0])

sp500.head(20)

# Start 1 Jan 2011
bitcoin = bitcoin.loc['2011-01-01':]
sp500 = sp500.loc['2011-01-01':]

# Visual check of price data
sp500['Close'].plot(figsize=(12,8))

(sp500['Close'].pct_change() * 100).hist(bins=100, figsize = (12,10), by = lambda x: x.year)

sp500 = pd.DataFrame(sp500['Close'])

# Join data using S&P500 dates
data = sp500.join(bitcoin)

data.columns = ['bitcoin','sp500']

data.head(20)

# Calculate returns from prices
data['sp500_ret'] = data['sp500'].pct_change()
data['bitcoin_ret'] = data['bitcoin'].pct_change()
data.dropna(inplace = True)

# Lets do it on weekly returns as well
weekly = data.resample('W').last()
weekly['sp500_ret'] = weekly['sp500'].pct_change()
weekly['bitcoin_ret'] = weekly['bitcoin'].pct_change()

# Chart rolling corr of daily returns and compare to rolling corr of weekly returns
title = "Rolling 1 year correlation between Bitcoin and S&P500"
fig, ax = plt.subplots()
data['sp500_ret'].rolling(window=252).corr(data['bitcoin_ret']).plot(figsize = (12,8), title = title, legend = True, ax = ax)
weekly['sp500_ret'].rolling(window=52).corr(weekly['bitcoin_ret']).plot(figsize = (12,8), legend = True, ax = ax)
ax.legend(["Daily Returns", "Weekly Returns"]);

fig, ax = plt.subplots()
title = "Rolling 1 year correlation between Bitcoin and S&P500"
weekly['sp500_ret'].rolling(window=52).corr(weekly['bitcoin_ret']).plot(figsize = (12,8), title = title, ax = ax, legend = True)
ax.legend(['Weekly returns'])

monthly = data.resample('M').last()
monthly['sp500_ret'] = monthly['sp500'].pct_change()
monthly['bitcoin_ret'] = monthly['bitcoin'].pct_change()
monthly.head(10)

fig, ax = plt.subplots()
title = "Rolling 1 year correlation between Bitcoin and S&P500"
monthly['sp500_ret'].rolling(window=12).corr(monthly['bitcoin_ret']).plot(figsize = (12,8), title = title, ax = ax, legend = True)
ax.legend(['Monthly returns'])

