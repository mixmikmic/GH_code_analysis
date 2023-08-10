import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
get_ipython().magic('matplotlib inline')

chi2_rv = sts.chi2(3)            # 3 degrees of freedom
sample = chi2_rv.rvs(1000)       # 1000 variates

plt.hist(sample, bins=40, normed=True, label='hist')     # histogram based on sample

x = np.linspace(0, 15)
pdf = chi2_rv.pdf(x)                                     # theorettical probability density function
plt.plot(x, pdf, label='theoretical pdf', alpha=0.5)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

def calc_mean(size):
    samples = []
    sample = None
    for i in range(0, 1000):
        sample = chi2_rv.rvs(size)                 # generate sample with size = size
        samples.append(np.mean(sample))            # append mean of the sample
    return samples                                 # return means for each sample

samples_mean = calc_mean(5)                                             # size = 10
hist5 = plt.hist(samples_mean, bins=25, normed=True, label='hist')      # histogram of means

# add theoretical dist to the plot
x = np.linspace(0, 10)
norm_rv = sts.norm(chi2_rv.mean(), np.sqrt(chi2_rv.var() / 5.))  
pdf = norm_rv.pdf(x)
plt.plot(x, pdf, label='theoretical pdf', alpha=1, color='r', linewidth=2.0)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

hist10 = plt.hist(calc_mean(10), bins=25, normed=True, label='hist')     # size = 100

norm_rv = sts.norm(chi2_rv.mean(), np.sqrt(chi2_rv.var() / 10.))
pdf = norm_rv.pdf(x)
plt.plot(x, pdf, label='theoretical pdf', alpha=1, color='r', linewidth=2.0)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

hist50 = plt.hist(calc_mean(50), bins=25, normed=True, label='hist')     # size = 500

norm_rv = sts.norm(chi2_rv.mean(), np.sqrt(chi2_rv.var() / 50.))         # k = 3
pdf = norm_rv.pdf(x)
plt.plot(x, pdf, label='theoretical pdf', alpha=1, color='r', linewidth=2.0)
plt.legend()
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

