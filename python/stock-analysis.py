import pandas as pd
import quandl
import pandas_datareader as pdr # Use to import data from the web 
# Package and modules for importing data; this code may change depending on pandas version
import datetime # use for start and end dates 
import matplotlib.pyplot as plt

# Use google to get stock data 
aapl = pdr.get_data_google('AAPL', start = datetime.datetime(2016, 5, 7), end = datetime.datetime(2017, 6, 7))
# stock, start, end 
aapl.tail()

# Using quandl to import data 
aapl_quandl = quandl.get("WIKI/AAPL", start_date = "2006-6-1", end_date = "2012-6-1") # YYYY-MM-DD, not enforced
aapl_quandl.head()

# aapl.describe()
# Look at last 10 observations of a column 
last_10_aapl = aapl['High'][-10:]
last_10_aapl

# Sample 20 rows 
sample = aapl.sample(20) # sample 20 rows 
monthly_aapl = aapl.resample('M').mean() # Resample data so that aapl is at monthly level, not daily. 
# Resample is used to 
(monthly_aapl) 

aapl['diff']= aapl.Open - aapl.Close
aapl.head()

# Plot the closing prices for 'aapl'
aapl['Close'].plot(grid = True)
plt.show()

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pylab
get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = (12, 6) # Larger plots
stock["Adj Close"].plot(grid = True) # adds grid lining 

microsoft = web.DataReader("MSFT", "yahoo", start, end)
google = web.DataReader("GOOG", "yahoo", start, end)
apple = web.DataReader("AAPL", "yahoo", start, end)

myStocks = ['AAPL', 'GOOG', 'MSFT']
myStock_data = [web.DataReader(stock, "yahoo", start, end) for stock in myStocks]

# Create a DataFrame consisting of the adjusted closing price of these stocks, 
# first by making a list of these objects and using the join method
stocks = pd.DataFrame({myStock : stock["Adj Close"], #compare to others 
                       "AAPL": apple["Adj Close"],
                       "MSFT": microsoft["Adj Close"],
                       "GOOG": google["Adj Close"]})
stocks.tail()

stocks.plot(grid = True, title="Absolute Prices") 

stocks.plot(secondary_y = ["AAPL", "MSFT", myStock], grid = True, title = "Relative Prices")

from IPython.display import display, Math, Latex
display(Math(r'return = \frac{price_t}{price_0}'))

# df.apply(arg) will apply the function arg to each column in df, and return a DataFrame with the result
# Recall that lambda x is an anonymous function accepting parameter x; in this case, x will be a pandas Series object
stock_return = stocks.apply(lambda x: x / x[0])
stock_return.head()

stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)

# Use google to get stock data 
snap = pdr.get_data_google('SNAP', start = datetime.datetime(2016, 4, 1), end = datetime.datetime(2017, 6, 5))
# stock, start, end 
snap.tail()



