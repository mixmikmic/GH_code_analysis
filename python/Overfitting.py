import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels import regression
from scipy import poly1d
import auquanToolbox.dataloader as dl

x = np.arange(10)
y = 2*np.random.randn(10) + x**2
xs = np.linspace(-0.25, 9.25, 200)

lin = np.polyfit(x, y, 1)
quad = np.polyfit(x, y, 2)
many = np.polyfit(x, y, 9)

plt.figure(figsize=(15,7))
plt.scatter(x, y)
plt.plot(xs, poly1d(lin)(xs))
plt.plot(xs, poly1d(quad)(xs))
plt.plot(xs, poly1d(many)(xs))
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(['Underfit', 'Good fit', 'Overfit']);
plt.show()

# Load one year's worth of pricing data for five different assets
start = '2014-01-01'
end = '2016-01-01'
m1='PEP'
m2='MCD'
m3 ='CVS'
m4='DOW'
m5='PG'
data = dl.load_data_nologs('nasdaq', [m1,m2,m3,m4,m5], start, end)
x1 = data['ADJ CLOSE'][m1]
x2 = data['ADJ CLOSE'][m2]
x3 = data['ADJ CLOSE'][m3]
x4 = data['ADJ CLOSE'][m4]
y = data['ADJ CLOSE'][m5]

# Build a linear model using only x1 to explain y
slr = regression.linear_model.OLS(y, sm.add_constant(x1)).fit()
slr_prediction = slr.params[0] + slr.params[1]*x1

# Run multiple linear regression using x1, x2, x3, x4 to explain y
mlr = regression.linear_model.OLS(y, sm.add_constant(np.column_stack((x1,x2,x3,x4)))).fit()
mlr_prediction = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4

# Compute adjusted R-squared for the two different models
print ('Using only PEP R-squared:', slr.rsquared_adj)
print ('Using only PEP p-value:', slr.f_pvalue)
print ('Using a basket of stocks R-squared:', mlr.rsquared_adj)
print ('Using a basket of stocks p-value:', mlr.f_pvalue)
# Plot y along with the two different predictions
y.plot(figsize=(15,7))
slr_prediction.plot()
mlr_prediction.plot()
plt.legend(['PG', 'Only PEP', 'Basket of stocks']);
plt.show()

# Load a year and a half of pricing data
start = '2016-01-01'
end = '2016-06-01'
data = dl.load_data_nologs('nasdaq', [m1,m2,m3,m4,m5], start, end)
x1 = data['ADJ CLOSE'][m1]
x2 = data['ADJ CLOSE'][m2]
x3 = data['ADJ CLOSE'][m3]
x4 = data['ADJ CLOSE'][m4]
y = data['ADJ CLOSE'][m5]

# Extend our model from before to the new time period
slr_prediction2 = slr.params[0] + slr.params[1]*x1
mlr_prediction2 = mlr.params[0] + mlr.params[1]*x1 + mlr.params[2]*x2 + mlr.params[3]*x3 + mlr.params[4]*x4

# Compute adjusted R-squared over the extended time period
adj = float(len(y) - 1)/(len(y) - 5) # Compute adjustment factor
SST = sum((y - np.mean(y))**2)
SSRs = sum((slr_prediction2 - y)**2)
print ('Using only PEP R-squared:', 1 - adj*SSRs/SST)
SSRm = sum((mlr_prediction2 - y)**2)
print ('Using a basket of stocks R-squared:', 1 - adj*SSRm/SST)

# Plot y along with the two different predictions
y.plot(figsize=(15,7))
slr_prediction2.plot()
mlr_prediction2.plot()
plt.legend(['PG', 'Only PEP', 'Basket']);
plt.show()

# Load the pricing data for a stock
start = '2012-01-01'
end = '2014-06-30'
assets = ['MCD']
data = dl.load_data_nologs('nasdaq', assets, start, end)
prices = data['ADJ CLOSE']
asset = prices.iloc[:, 0]
# Compute rolling averages for various window lengths
mu_30d = asset.rolling(window=30, center=False).mean()
mu_60d = asset.rolling(window=60, center=False).mean()
mu_100d = asset.rolling(window=100, center=False).mean()

# Plot asset pricing data with rolling means from the 100th day, when all the means become available
plt.figure(figsize=(15,7))
plt.plot(asset[100:], label='Asset')
plt.plot(mu_30d[100:], label='30d MA')
plt.plot(mu_60d[100:], label='60d MA')
plt.plot(mu_100d[100:], label='100d MA')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()

# Trade using a simple mean-reversion strategy
def trade(stock, length):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if length == 0:
        return 0
    
    # Compute rolling mean and rolling standard deviation
    mu = stock.rolling(window=length, center=False).mean()
    std = stock.rolling(window=length, center=False).std()
    
    # Compute the z-scores for each day using the historical data up to that day
    zscores = (stock - mu)/std
    
    # Simulate trading
    # Start with no money and no positions
    money = 0
    count = 0
    for i in range(len(stock)):
        # Sell short if the z-score is > 1
        if zscores[i] > 1:
            money += stock[i]
            count -= 1
        # Buy long if the z-score is < 1
        elif zscores[i] < -1:
            money -= stock[i]
            count += 1
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscores[i]) < 0.5:
            money += count*stock[i]
            count = 0
    return money

# Find the window length 0-254 that gives the highest returns using this strategy
length_scores = [trade(asset, l) for l in range(255)]
best_length = np.argmax(length_scores)
print ('Best window length:', best_length)

# Get pricing data for a different timeframe
start2 = '2014-06-30'
end2 = '2017-01-01'
assets = ['MCD']
data2 = dl.load_data_nologs('nasdaq', assets, start2, end2)
prices2 = data2['ADJ CLOSE']
asset2 = prices2.iloc[:, 0]

# Find the returns during this period using what we think is the best window length
length_scores2 = [trade(asset2, l) for l in range(255)]
print (best_length, 'day window:', length_scores2[best_length])

# Find the best window length based on this dataset, and the returns using this window length
best_length2 = np.argmax(length_scores2)
print (best_length2, 'day window:', length_scores2[best_length2])

plt.figure(figsize=(15,7))
plt.plot(length_scores)
plt.plot(length_scores2)
plt.xlabel('Window length')
plt.ylabel('Score')
plt.legend(['2012-2014', '2014-2016'])
plt.show()

