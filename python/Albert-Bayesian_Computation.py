get_ipython().magic('matplotlib inline')
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import scipy.stats as stats

def tstat(x, y):
    m, n = len(x), len(y)
    
    # pooled standard deviation
    sp = np.sqrt(((m-1)*np.std(x)**2 + (n-1)*np.std(y)**2)/(m+n-2))
    return (np.mean(x) - np.mean(y)) / (sp * np.sqrt(1./m + 1./n))

x = stats.norm.rvs(loc=50,scale=10,size=10000)
y = stats.norm.rvs(loc=50,scale=100,size=10000)
x = random.randn(100000)*1 + 50
y = random.randn(100000)*1 + 50

print(tstat(x,y))

# scipy's version, using Welch, which does not assume equal variance
print(stats.ttest_ind(x, y, equal_var=False)[0])
print(stats.ttest_rel(x, y)[0])

np.std(x)

stats.ttest_ind(x, y, equal_var=False)

