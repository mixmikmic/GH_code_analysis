import pandas as pd
from pandas_datareader import data
# pandas uses this to plot
import matplotlib.pyplot as plt

# By default matplotlib would plot outside of the notebook. We need to fix that so that it renders inline
get_ipython().magic('matplotlib inline')

# Make the plots larger
plt.rcParams['figure.figsize'] = (15.0, 7.5)

# Get stock information for Blackberry
bb = data.DataReader(name = 'BBRY', data_source='google', start='2007-07-01', end='2008-12-31')
bb.head()

# Calling plot directly on the dataframe:
bb.plot()

# X axis is the months, which is the dataframe index

# y -> Name of column(s) to plot
bb.plot(y='Volume')

bb.plot(y='High')

# Can also call plot directly on a series
bb['Close'].plot()

# Plots two columns
bb[['High', 'Low']].plot()

# Available templates
plt.style.available

# Default plot
bb.plot(y = "Close")

# Change style
# **THIS CHANGES IT FOR THE ENTIRE NOTEBOOK**
plt.style.use('fivethirtyeight')
bb.plot(y = "Close")

plt.style.use('dark_background')
bb.plot(y = "Close")

plt.style.use('ggplot')
bb.plot(y = "Close")

google = data.DataReader(name = 'GOOG', data_source = 'google', start='2004-01-01', end='2016-12-31')
google.head()

# Rank performance based on our logic
#   < 200 = Poor
#   200 < x < 500 = Satisfactory
#   > 500 = Excellent
def rank_performance(stock_price):
    if stock_price <= 200:
        return "Poor"
    elif stock_price > 200 and stock_price <= 500:
        return "Satisfactory"
    else:
        return "Excellent"

# A bar chart would work perfectly for showing how value counts

# kind - Default is 'line'; Want 'bar' to get a bar chart
google['Close'].apply(rank_performance).value_counts().plot(kind = 'bar')

# Horizontal barchart - kind = barh
google['Close'].apply(rank_performance).value_counts().plot(kind = 'barh')

apple = data.DataReader(name = 'AAPL', data_source = 'google', start='2012-01-01', end='2016-12-31')
apple.head()

# Compare each day's close price to average
stock_mean = apple['Close'].mean()
stock_mean

def rank_performance(stock_price):
    if stock_price >= stock_mean:
        return "Above"
    else:
        return "Below"

apple['Close'].apply(rank_performance).value_counts()

# Plot the above as a pie chart
apple['Close'].apply(rank_performance).value_counts().plot(kind='pie')

# Legend is missing in the above. Fix by setting 'legend' = True
apple['Close'].apply(rank_performance).value_counts().plot(kind='pie', legend=True)

google = data.DataReader(name = 'GOOG', data_source = 'google', start='2004-01-01', end='2016-12-31')
google.head()

# Create buckets of Googles stock prices in buckets of 100 

# This function rounds down
def custom_round(stock_price):
    return int(stock_price / 100.0) * 100

google['High'].apply(custom_round).value_counts().sort_index()

google['High'].apply(custom_round).nunique()

google['High'].apply(custom_round).plot(kind='hist', bins=google['High'].apply(custom_round).nunique())

