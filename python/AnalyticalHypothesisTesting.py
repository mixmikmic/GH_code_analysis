get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp

# this is our dataset
x = [1.0, 2.0, -.5, 3.0, 4.0, .025, .25, .5, 7, 3.3, 1.9]
plt.plot(x, [0]*len(x), 'b.')
plt.xlabel('x')
plt.xlim([-5, 5])
plt.show()
ttest_1samp(x, 0)

def get_t_stat(x):
    """ Computes the t-statistic for the specified sample x """
    return len(x)**0.5*np.mean(x) / (np.std(x, ddof=1))

t_stat = get_t_stat(x)
print t_stat

n_trials = 100000
s = np.std(x, ddof=1)
p_value = np.mean([abs(get_t_stat(np.random.normal(scale=sigma,
                                                   size=(len(values),)))) > t_stat
                   for i in range(n_trials)])
print "p-value computed via simulation", p_value

import seaborn as sns

t_stats = [get_t_stat(np.random.normal(scale=sigma,
                                       size=(len(values),))) for i in range(n_trials)]
sns.distplot(t_stats)

import scipy.stats

sns.distplot(t_stats)
points = np.arange(plt.xlim()[0], plt.xlim()[1], .1)
plt.plot(points, [scipy.stats.t.pdf(point, len(x) - 1) for point in points])
plt.show()

1 - scipy.stats.t.cdf(t_stat, len(x) - 1) + scipy.stats.t.cdf(-t_stat, len(x) - 1)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import linregress, pearsonr

def get_t_stat_corr(v):
    """ Computes the t-statistic for testing the correlation between
        the two columns of v """
    x = v[:,0]
    y = v[:,1]
    N = v.shape[0]
    R = np.corrcoef(x,y)[0, 1]
    return R / sqrt((1 - R**2) / (N - 2))

# make a sample
data = np.asarray([[1.0, .5],
                   [0.25, .1],
                   [0.5, .4], 
                   [0.6, .3],
                   [.7, .7],
                   [0.8, .7],
                   [.4, .2],
                   [1.0, 1.0]])

plt.scatter(data[:,0], data[:,1])
plt.show()

# compute the t-statistic for our sample
t_stat = get_t_stat_corr(data)
print "actual t_stat", t_stat

# compute the p-value using the analytical approach
_, p_value_analytical = pearsonr(data[:,0], data[:,1])
print "analytical p-value is", p_value_analytical

n_trials = 100000

t_stats = [get_t_stat_corr(np.random.normal(size=data.shape)) for i in range(n_trials)]
print "simulation p value", np.mean(np.abs(t_stats) > t_stat)

sns.distplot(t_stats)
points = np.arange(plt.xlim()[0], plt.xlim()[1], .1)
plt.plot(points, [scipy.stats.t.pdf(point, len(x) - 2) for point in points])
plt.show()

1 -     scipy.stats.t.cdf(t_stat, data.shape[0] - 2) +     scipy.stats.t.cdf(-t_stat, data.shape[0] - 2)

