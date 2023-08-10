import pandas as pd
get_ipython().magic('matplotlib inline')

close_px_all = pd.read_csv('stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B').ffill()
close_px.head()

close_px['AAPL'].plot()

close_px.ix['2009'].plot()

close_px['AAPL'].ix['01-2011':'03-2011'].plot()

# http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.Series.resample.html
appl_q = close_px['AAPL'].resample('Q-DEC').ffill() #['M', 'A', 'Q', 'BM', 'BA', 'BQ', 'W'])

appl_q.ix['2009':].plot()

close_px.AAPL.plot()
close_px.AAPL.rolling(window=250, center=False).mean().plot()

appl_std250 = close_px.AAPL.rolling(window=250, min_periods=10, center=False).std()
appl_std250[5:12]

appl_std250.plot()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# two subgraphs
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12, 7))

# data
aapl_px = close_px.AAPL['2005':'2009']
ma60 = aapl_px.rolling(window=60, min_periods=50,center=False).mean()
ewma60 = aapl_px.ewm(span=60,min_periods=0,adjust=True,ignore_na=False).mean()

# plot 0
aapl_px.plot(style='k-', ax=axes[0])
ma60.plot(style='k--', ax=axes[0])
axes[0].set_title('Simple MA')

# plot 1
aapl_px.plot(style='k-', ax=axes[1])
ewma60.plot(style='k--', ax=axes[1])
axes[1].set_title('Exponentially-weighted MA')

