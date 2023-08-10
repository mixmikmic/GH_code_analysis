import numpy as np
import matplotlib.pyplot as plt

# Get returns data for S&P 500
start = '2014-01-01'
end = '2015-01-01'
spy = get_pricing('SPY', fields='price', start_date=start, end_date=end).pct_change()[1:]

# Plot a histogram using 20 bins
fig = plt.figure(figsize = (16, 7))
_, bins, _ = plt.hist(spy, 20)
labels = ['%.3f' % a for a in bins] # Reduce precision so labels are legible
plt.xticks(bins, labels)
plt.xlabel('Returns')
plt.ylabel('Number of Days')
plt.title('Frequency distribution of S&P 500 returns, 2014');

# Example of a cumulative histogram
fig = plt.figure(figsize = (16, 7))
_, bins, _ = plt.hist(spy, 20, cumulative='True')
labels = ['%.3f' % a for a in bins]
plt.xticks(bins, labels)
plt.xlabel('Returns')
plt.ylabel('Number of Days')
plt.title('Cumulative distribution of S&P 500 returns, 2014');

# Get returns data for some security
asset = get_pricing('MSFT', fields='price', start_date=start, end_date=end).pct_change()[1:]

# Plot the asset returns vs S&P 500 returns
plt.scatter(asset, spy)
plt.xlabel('MSFT')
plt.ylabel('SPY')
plt.title('Returns in 2014');

spy.plot()
plt.ylabel('Returns');

