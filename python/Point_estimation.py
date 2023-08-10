from IPython.display import YouTubeVideo
YouTubeVideo('dR4a4jYHAWI')

from __future__ import print_function, division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
get_ipython().magic('matplotlib inline')

# Drawing samples from a Gaussian of mean 0, stdev 1
max_sample = 100
np.random.seed(100)
gaussian_sample = sp.stats.norm.rvs(size=max_sample)

bias_var = []
unbias_var = []
for N in range(5,max_sample+1):
    test_sample = gaussian_sample[:N]
    test_mean = test_sample.mean()
    bvar = (1/N)*np.sum((test_sample-test_mean)**2)
    ubvar = (1/(N-1))*np.sum((test_sample-test_mean)**2)
    bias_var.append(bvar)
    unbias_var.append(ubvar)
bias_var = np.array(bias_var)
unbias_var = np.array(unbias_var)
bias_std = np.sqrt(bias_var)
unbias_std = np.sqrt(unbias_var)

plt.plot(np.arange(5,max_sample+1), bias_std,
         color='r', linestyle='--', linewidth=2, label='Biased')
plt.plot(np.arange(5,max_sample+1), unbias_std,
         color='k', linewidth=2, label='Unbiased')
plt.xlabel('N', fontsize=20)
plt.ylabel('StDev', fontsize=20)
plt.legend(fontsize=12)
plt.tick_params(labelsize=14)
plt.title('Stdev of a Gaussian', fontsize=20)
sns.despine()

x_robustness = sp.stats.norm.rvs(size=100)
x_robustness[10] = 1.0*1e7  # A wild scraper appears!
print('X mean: %.3f' % x_robustness.mean())
print('X median: %.3f' % np.median(x_robustness))  # It's super effective!

cauchy_sample_size=1000
num_runs = 100
np.random.seed(200)

cauchy_mean = np.zeros(num_runs)
cauchy_median = np.zeros(num_runs)

for ind in range(num_runs):
    cauchy_sample = sp.stats.cauchy.rvs(size=cauchy_sample_size)
    cauchy_mean[ind] = cauchy_sample.mean()
    cauchy_median[ind] = np.median(cauchy_sample)

f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
ax1.hist(cauchy_mean, normed=True, bins=40, color='k', alpha=0.3, label='Mean')
ax2.hist(cauchy_median, normed=True, bins=10, color='r', alpha=0.5, label='Median')
ax1.set_title('Cauchy Mean', fontsize=20)
ax2.set_title('Cauchy Median', fontsize=20)
# plt.legend(fontsize=12)
# plt.tick_params(labelsize=14)
# plt.title('Finding the Middle of a Cauchy', fontsize=20)
sns.despine()

print('Stdev of mean: %.4f' % cauchy_mean.std())
print('Stdev of median: %.4f' % cauchy_median.std())

# However, for a normally distributed sample with no outliers,
# the mean is more efficient.

norm_sample_size=100
num_runs = 100
np.random.seed(200)

norm_mean = np.zeros(num_runs)
norm_median = np.zeros(num_runs)

for ind in range(num_runs):
    norm_sample = sp.stats.norm.rvs(size=norm_sample_size)
    norm_mean[ind] = norm_sample.mean()
    norm_median[ind] = np.median(norm_sample)
    
f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
ax1.hist(norm_mean, normed=True, bins=20, color='k', alpha=0.3, label='Mean')
ax2.hist(norm_median, normed=True, bins=20, color='r', alpha=0.5, label='Median')
ax1.set_title('Norm Mean', fontsize=20)
ax2.set_title('Norm Median', fontsize=20)
# plt.legend(fontsize=12)
# plt.tick_params(labelsize=14)
# plt.title('Finding the Middle of a Cauchy', fontsize=20)
sns.despine()

print('Stdev of mean: %.4f' % norm_mean.std())
print('Stdev of median: %.4f' % norm_median.std())



