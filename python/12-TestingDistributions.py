# Imports
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import normaltest

# Create a dataset of normally distributed data
d1 = stats.norm.rvs(size=100000)

# Plot a histogram of the observed data
#  Included is expected distribution, if the data is normally distributed, with the same mean and std of the data. 
xs = np.arange(d1.min(), d1.max(), 0.1)
fit = stats.norm.pdf(xs, np.mean(d1), np.std(d1))
plt.plot(xs, fit, label='Normal Dist.', lw=3)
plt.hist(d1, 50, normed=True, label='Actual Data');
plt.legend();

# In scipy, the 'normaltest' function tests whether a sample differs from a normal distribution
#  The null hypothesis is that the data are normally distributed.
#    We can use normaltest to check this null - do we have to reject the null (to claim the data are not normal).
#  It does using a combined statistics comparing the skew and kurtosis of the observed
#    data, as compared to as expected under a normal distribution. 
get_ipython().magic('pinfo normaltest')

# Run normal test on the data
stat, p = normaltest(d1)

# Check the p-value of the normaltest
print('\nNormaltest p value is: ', p, '\n')

# With alpha value of 0.05, how should we proceed
if p < 0.05:
    print('We have evidence to reject the null hypothesis, that the data are normally distributed.')
else:
    print('We do not have evidence to reject the null hypothesis.')

# Generate some data from a beta distribution
d2 = stats.beta.rvs(7, 10, size=100000)

# Plot a histogram of the observed data
#  Included is expected distribution, if the data is normally distributed, with the same mean and std of the data. 
xs = np.arange(d2.min(), d2.max(), 0.01)
fit = stats.norm.pdf(xs, np.mean(d2), np.std(d2))
plt.plot(xs, fit, label='Normal Dist.', lw=3)
plt.hist(d2, 50, normed=True, label='Actual Data');
plt.legend();

# Note that we can see *some* differences, when plotting the PDF
#  However, if you turn off the PDF plot, we might guess these data look pretty normal

# Run normal test on the data
stat, p = normaltest(d2)

# Check the p-value of the normaltest
print('\nNormaltest p value is: ', p, '\n')

# With alpha value of 0.05, how should we proceed
if p < 0.05:
    print('We have evidence to reject the null hypothesis, that the data are normally distributed.')
else:
    print('We do not have evidence to reject the null hypothesis.')

from scipy.stats import kstest

get_ipython().magic('pinfo kstest')

