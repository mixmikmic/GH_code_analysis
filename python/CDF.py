get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from scipy.stats import norm


# Get random samples
n = 2000
#X=np.random.random_sample(n)

# Get a normal distribution
X = norm.rvs(size=n) 

# Sort data
X = np.sort(X)



# The scipy functions are useful if your distribution can be fitted to a known distribution.
# Get PDF function:
rv = norm()
PDF_normal = rv.pdf(X)
CDF_normal = rv.cdf(X)
# The above will produce the CDF for a normal distribution, so we need this instead:


# If you don't know the shape of the distribution:

# Solution 1:
# This assumes that there are no repeated values:
CDF1 = 1. * np.arange(len(X)) / (len(X) - 1)

# Solution 2: This solution is probably overkilling, 
# but this works in the practice:
# Percentiles is an array from 0 to 100:
percentiles = np.arange(0,1,(1.0-0.0)/len(X))*100.0
CDF=[]
for percentile in percentiles:
    CDF.append([percentile/100.0,np.percentile(X,percentile)])
CDF=np.array(CDF)
# Note this CDF contains 2 columns: 
# 1. percentile and 2. the corresponding value.
CDF

import matplotlib.pyplot as plt

xstep=0.01

fig, ax = plt.subplots()


ax.plot(X, PDF_normal, 'k-', lw=2, label='PDF for normal D')
ax.plot(X, CDF_normal, 'g.', lw=5, alpha=0.6, label = 'CDF for normal D' )
#
# CDF's obtained from more general approaches:
ax.plot(X, CDF1, 'r+', lw=5, alpha=0.6, label = 'CDF 1' )
ax.plot(CDF[:,1],CDF[:,0], 'y-', lw=5, alpha=0.6, label = 'CDF 2' )


n,bins,patches=ax.hist(X, bins=np.arange(X.min(), X.max(), xstep),normed=1,facecolor='blue',align='mid',label='distribution')


ax.legend(loc='right')
#ax.set_title('PDF function')
ax.set_ylabel('CDF')
ax.set_xlabel('x')


plt.show()



