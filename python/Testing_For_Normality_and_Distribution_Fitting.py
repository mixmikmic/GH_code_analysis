get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
print(plt.style.available)

plt.style.use('ggplot')
data = []
for i in range(1000):
    data.append(random.normalvariate(0,1))

plt.hist(data)
plt.show()

stats.probplot(data, dist=stats.norm, sparams=(0,1), plot=plt)
plt.show()

_ , pvalue = stats.normaltest(data)
if pvalue > 0.05:
    print("Data is most likely from Normal distribution")

pVals = pd.Series()
# The scipy normaltest is based on D-Agostino and Pearsons test that
# combines skew and kurtosis to produce an omnibus test of normality.
_, pVals['omnibus'] = stats.normaltest(data)

# Shapiro-Wilk test
_, pVals['Shapiro-Wilk'] = stats.shapiro(data)
    
# Or you can check for normality with Lilliefors-test
ksStats, pVals['Lilliefors'] = kstest_normal(data)
    
# Alternatively with original Kolmogorov-Smirnov test
_, pVals['KS'] = stats.kstest((data-np.mean(data))/np.std(data,ddof=1), 'norm')

print(pVals)

stats.probplot(data, plot=plt)
plt.show()

stats.normaltest(data)

# Import standard packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

# additional packages
from statsmodels.stats.diagnostic import kstest_normal

myMean = 0
mySD = 3
x = np.arange(-5,15,0.1)

def check_normality():
    '''Check if the distribution is normal.'''
    # Generate and show a distribution
    numData = 100
    
    # To get reproducable values, I provide a seed value
    np.random.seed(987654321)   
    
    data = stats.norm.rvs(myMean, mySD, size=numData)
    plt.hist(data)
    plt.show()

    # --- >>> START stats <<< ---
    # Graphical test: if the data lie on a line, they are pretty much
    # normally distributed
    _ = stats.probplot(data, plot=plt)
    plt.show()

    pVals = pd.Series()
    # The scipy normaltest is based on D-Agostino and Pearsons test that
    # combines skew and kurtosis to produce an omnibus test of normality.
    _, pVals['omnibus'] = stats.normaltest(data)

    # Shapiro-Wilk test
    _, pVals['Shapiro-Wilk'] = stats.shapiro(data)
    
    # Or you can check for normality with Lilliefors-test
    ksStats, pVals['Lilliefors'] = kstest_normal(data)
    
    # Alternatively with original Kolmogorov-Smirnov test
    _, pVals['KS'] = stats.kstest((data-np.mean(data))/np.std(data,ddof=1), 'norm')
    
    print(pVals)
    if pVals['omnibus'] > 0.05:
        print('Data are normally distributed')
    # --- >>> STOP stats <<< ---
    
    return pVals['KS']
    
if __name__ == '__main__':
    p = check_normality()   

import matplotlib.pyplot as plt
import scipy
import scipy.stats
size = 20000
x = scipy.arange(size)
# creating the dummy sample (using beta distribution)
y = scipy.int_(scipy.round_(scipy.stats.beta.rvs(6,2,size=size)*47))
# creating the histogram
h = plt.hist(y, bins=range(48), normed=True)

dist_names = ['gamma', 'beta', 'rayleigh', 'norm']

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0,47)
plt.legend(loc='upper left')
plt.show()

dist = getattr(scipy.stats, 'norm')

param = dist.fit(y)
param

m = [0,1,2,3,4]

m[-1]

param[:-2]

from scipy.stats import norm
from numpy import linspace
import matplotlib.pyplot as plt

# picking 150 of from a normal distrubution
# with mean 0 and standard deviation 1
samp = norm.rvs(loc=0,scale=1,size=1000) 

param = norm.fit(samp) # distribution fitting

# now, param[0] and param[1] are the mean and 
# the standard deviation of the fitted distribution
x = linspace(-5,5,100)
# fitted distribution
pdf_fitted = norm.pdf(x,loc=param[0],scale=param[1])
# original distribution
pdf = norm.pdf(x)

plt.title('Normal distribution')
plt.plot(x,pdf_fitted,'r-',x,pdf,'b-')
plt.hist(samp,normed=True)
plt.show()

import pandas as pd

df = pd.read_excel(r'D:\temp\RDX_Battery.xlsx', 'Claims')

dtf = df['DAYS_TO_FAIL_MINZERO']

dtf.hist()

dtf.describe()



stats.probplot(dtf.values, dist=stats.lognorm, sparams=(0,1), plot=plt)
plt.show()

