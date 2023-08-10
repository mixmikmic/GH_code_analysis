import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats

TRUE_MEAN = 40
TRUE_STD = 10
X = numpy.random.normal(TRUE_MEAN, TRUE_STD, 1000)

def normal_mu_MLE(X):
    # Get the number of observations
    T = len(data)
    # Sum the observations
    s = sum(X)
    return 1.0/T * s

def normal_sigma_MLE(X):
    T = len(X)
    # Get the mu MLE
    mu = normal_mu_mle(X)
    # Sum the square of the differences
    s = sum( math.pow((X - mu), 2) )
    # Compute sigma^2
    sigma_squared = 1.0/T * s
    return math.sqrt(sigma_squared)

print "Mean Estimation"
print normal_mu_mle(X)
print np.mean(X)
print "Standard Deviation Estimation"
print normal_sigma_mle(X)
print np.std(X)

mu, std = scipy.stats.norm.fit(X)
print "mu estimate: " + str(mu)
print "std estimate: " + str(std)

pdf = scipy.stats.norm.pdf
# We would like to plot our data along an x-axis ranging from 0-80 with 80 intervals
# (increments of 1)
x = np.linspace(0, 80, 80)
h = plt.hist(X, bins=x, normed='true')
l = plt.plot(pdf(x, loc=mu, scale=std))

TRUE_LAMBDA = 5
X = np.random.exponential(TRUE_LAMBDA, 1000)

def exp_lamda_MLE(X):
    T = len(X)
    s = sum(X)
    return s/T

print "lambda estimate: " + str(exp_lamda_MLE(X))

# The scipy version of the exponential distribution has a location parameter
# that can skew the distribution. We ignore this by fixing the location
# parameter to 0 with floc=0
_, l = scipy.stats.expon.fit(X, floc=0)

pdf = scipy.stats.expon.pdf
x = range(0, 80)
h = plt.hist(X, bins=x, normed='true')
l = plt.plot(pdf(x, scale=l))

prices = get_pricing('TSLA', fields='price', start_date='2014-01-01', end_date='2015-01-01')
# This will give us the number of dollars returned each day
absolute_returns = np.diff(prices)
# This will give us the percentage return over the last day's value
# the [:-1] notation gives us all but the last item in the array
# We do this because there are no returns on the final price in the array.
returns = absolute_returns/prices[:-1]

mu, std = scipy.stats.norm.fit(returns)
pdf = scipy.stats.norm.pdf
x = np.linspace(-1,1, num=100)
h = plt.hist(returns, bins=x, normed='true')
l = plt.plot(x, pdf(x, loc=mu, scale=std))

