# Import libraries
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

# Get data for the specified period and stocks
start = '2014-01-01'
end = '2015-01-01'
asset = get_pricing('TSLA', fields='price', start_date=start, end_date=end)
benchmark = get_pricing('SPY', fields='price', start_date=start, end_date=end)

# We have to take the percent changes to get to returns
# Get rid of the first (0th) element because it is NAN
r_a = asset.pct_change()[1:]
r_b = benchmark.pct_change()[1:]

# Let's plot them just for fun
r_a.plot()
r_b.plot();

# Let's define everything in familiar regression terms
X = r_b.values # Get just the values, ignore the timestamps
Y = r_a.values

# We add a constant so that we can also fit an intercept (alpha) to the model
# This just adds a column of 1s to our data
X = sm.add_constant(X)
model = regression.linear_model.OLS(Y, X)
model = model.fit()
# Remove the constant now that we're done
X = X[:, 1]
alpha = model.params[0]
beta = model.params[1]
print 'alpha: ' + str(alpha)
print 'beta: ' + str(beta)

X2 = np.linspace(X.min(), X.max(), 100)
Y_hat = X2 * beta + alpha

plt.scatter(X, Y, alpha=0.3) # Plot the raw data
plt.xlabel("r_b")
plt.ylabel("r_a")

# Add the regression line, colored in red
plt.plot(X2, Y_hat, 'r', alpha=0.9);

# Use numpy to find the standard deviation of the returns
SD = np.std(Y)
print SD

# Let's compute the volatility for our benchmark, as well
benchSD = np.std(X)
print benchSD

vol = SD*(252**.5)
print vol

benchvol = benchSD*(252**.5)
print benchvol

# Since we have many distinct values, we'll lump them into buckets of size .02
x_min = int(math.floor(X.min()))
x_max = int(math.ceil(X.max()))
plt.hist(X, bins=[0.02*i for i in range(x_min*50, x_max*50)], alpha=0.5, label='S&P')
plt.hist(Y, bins=[0.02*i for i in range(x_min*50, x_max*50)], alpha=0.5, label='Stock')
plt.legend(loc='upper right');

# Get the returns for a treasury-tracking ETF to be used in the Sharpe ratio
# Note that BIL is only being used in place of risk free rate,
# and should not be used in such a fashion for strategy development
riskfree = get_pricing('BIL', fields='price', start_date=start, end_date=end)
r_b_S = riskfree.pct_change()[1:]
X_S = r_b_S.values

# Compute the Sharpe ratio for the asset we've been working with
SR = np.mean(Y - X_S)/np.std(Y - X_S)

# Compute the information ratio for the asset we've been working with, using the S&P index
IR = np.mean(Y - X)/np.std(Y - X)

# Print results
print 'Sharpe ratio: ' + str(SR)
print 'Information ratio: ' + str(IR)

# To compute the semideviation, we want to filter out values which fall above the mean
meandif = np.mean(Y - X_S)
lows = [e for e in Y - X_S if e <= meandif]

# Because there is no built-in semideviation, we'll compute it ourselves
def dist(x):
    return (x - meandif)**2
semidev = math.sqrt(sum(map(dist,lows))/len(lows))

Sortino = meandif/semidev
print 'Sortino ratio: ' + str(Sortino)

