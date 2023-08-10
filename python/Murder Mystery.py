import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid")

# Since there is a pretty low chance that a person will choose to buy a product
# Let us first create a random variable called buy.
# This is the variable whose value we are uncertain about
# This variable can take two values `true` and `false`
# P(buy = true) = 0.2 and P(buy = false) = 0.8
# This is a bernoulli distribution Bernoulli(buy|p), where p can take the value 0.2 or 0.8

from scipy.stats import bernoulli

p = 0.2
sample_size=10

rvs = bernoulli.rvs(p, size=sample_size)

# What proportion of samples are true ?
print 'Proportion of samples that are true is %f when sample size %d ' %(rvs.sum() * 1. / len(rvs), sample_size) 

sample_size=100

rvs = bernoulli.rvs(p, size=sample_size)

# What proportion of samples are true ?
print 'Proportion of samples that are true is %f when sample size %d ' %(rvs.sum() * 1. / len(rvs), sample_size) 

sample_size=1000

rvs = bernoulli.rvs(p, size=sample_size)

# What proportion of samples are true ?
print 'Proportion of samples that are true is %f when sample size %d ' %(rvs.sum() * 1. / len(rvs), sample_size) 

sample_size=10000

rvs = bernoulli.rvs(p, size=sample_size)

# What proportion of samples are true ?
print 'Proportion of samples that are true is %f when sample size %d ' %(rvs.sum() * 1. / len(rvs), sample_size) 



