#import libraries
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# For reading stock data from yahoo
import pandas_datareader as pdr
from datetime import datetime
from __future__ import division

import pandas_datareader.data as web
tech_list = ['AAPL','GOOG','MSFT','AMZN']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1,end.month,end.day)

for stock in tech_list:   
    globals()[stock] = web.DataReader(stock,'yahoo',start,end)    

#Descriptive & Exploratory analysis
AAPL.describe()

#DataFrame Structure
AAPL.info()

# Let's see historical view of the closing price for the last one year
AAPL['Adj Close'].plot(legend=True,figsize=(10,4))

# Now let's plot the total volume of stock being traded each day over the past year
AAPL['Volume'].plot(legend=True,figsize=(10,4))

#Calculate and plot the simple moving average (SMA) for upto 10, 20 and 50 days
moving_avg_day = [10,20,50]

for ma in moving_avg_day:
    column_name = "MA for %s days" % (str(ma))
    AAPL[column_name] =pd.rolling_mean(AAPL['Adj Close'],ma)
    
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(15,6))

#Now lets analyse the Daily Return Analysis as our first step to determing the volatility of the stock prices

# We'll use pct_change to find the percent change for each day
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
# Then we'll plot the daily return percentage
AAPL['Daily Return'].plot(figsize=(12,4),legend=True,linestyle='--',marker='o')

#Average daily return
#Using dropna to eliminate NaN values
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='red')

#Now lets do a comparative study to analyze the returns of all the stocks in our list

# Grab all the closing prices for the tech stocks into one new DataFrame
closing_df = web.DataReader(['AAPL','GOOG','MSFT','AMZN'],'yahoo',start,end)['Adj Close'] 

# Let's take a quick look
closing_df.head()

# Make a new DataFrame for storing stock daily return
tech_rets = closing_df.pct_change()

# Using joinplot to compare the daily returns of Google and Microsoft
sns.jointplot('GOOG','MSFT',tech_rets,kind='scatter')

# We can also simply call pairplot on our DataFrame for visual analysis of all the comparisons
sns.pairplot(tech_rets.dropna())

#Another analysis on the closing prices for each stock and their individual comparisons
# Call PairPLot on the DataFrame
close_fig = sns.PairGrid(closing_df)

# Using map_upper we can specify what the upper triangle will look like.
close_fig.map_upper(plt.scatter,color='purple')

# We can also define the lower triangle in the figure, including the plot type (kde) or the color map (BluePurple)
close_fig.map_lower(sns.kdeplot,cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the closing price
close_fig.map_diag(plt.hist,bins=30)

#For a quick look at the correlation analysis for all stocks, we can plot for the daily returns using corrplot
from seaborn.linearmodels import corrplot as cor
cor(tech_rets.dropna(),annot=True)

#It is important to note the risk involved with the expected return for each stock visually before starting the analysis
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)

# Setting the x and y limits & plt axis titles
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.004])
plt.xlabel('Expected returns')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))

rets['AAPL'].quantile(0.05)

# Set up time horizon and delta values
days = 365
dt = 1/days

# Calculate mu (drift) from the expected return data we got for AAPL
mu = rets.mean()['GOOG']

# Calculate volatility of the stock from the std() of the average return
sigma = rets.std()['GOOG']

def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation, mu, sigma and returns simulated price array'''

    price = np.zeros(days)
    price[0] = start_price    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Calculating and returning price array for number of days
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
    return price

# Get start price from GOOG.head()
start_price = 560.85
simulations = np.zeros(100)

for run in range(100):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Google')

# Now we'lll define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Now let's plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');



