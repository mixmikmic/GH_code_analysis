# %load ./snippets/data-imports
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

get_ipython().magic('matplotlib inline')

# generate boundaries, and infer low/high values if not provided
def infer_boundaries(low=None, high=None, mean=0, stddev=1):
    # generate the boundaries
    lowerBound = mean - 3 * stddev
    upperBound = mean + 3 * stddev
    
    # substitute for values not provided
    if(low is None):
        low = lowerBound
    if(high is None):
        high = upperBound        

    return (lowerBound, low, high, upperBound)

# display the % of area under normal curve between two values
def print_percent_coverage(low=None, high=None, mean=0, stddev=1):
    lowerBound, low, high, upperBound = infer_boundaries(low, high, mean, stddev)
    
    # print the % coverage
    pctiles = stats.norm.cdf([low, high], loc=mean, scale=stddev)
    area = (pctiles[1] - pctiles[0]) * 100
    print("The region between z-scores %.2f and %.2f covers %.2f%% of the area under the curve." % (low, high, area))    

# plot a normal curve and a filled area between 
# two values under the curve N(mean, stddev)
def plot_normal_curve(low=None, high=None, mean=0, stddev=1):
    lowerBound, low, high, upperBound = infer_boundaries(low, high, mean, stddev)
    
    # plot the curve
    x = np.linspace(lowerBound, upperBound, 1000)
    y = stats.norm.pdf(x, loc=mean, scale=stddev)
    curve = DataFrame({"x": x, "y": y})
    plt.plot(curve.x, curve.y)
    
    # plot the fill
    fill = curve[(curve.x >= low) & (curve.x <= high)]
    plt.fill_between(fill.x, fill.y, facecolor='darkorange', alpha=0.75)

    print_percent_coverage(low, high, mean, stddev)

plot_normal_curve(low=-1, high=1.5)

plot_normal_curve(high=1.8)

plot_normal_curve(low=-1)

z = stats.norm.ppf([0.9])[0]
z * 100 + 500

# percentiles => z-score : stats.norm.ppf
# z-score => percentiles : stats.norm.cdf
# z-score => height of normal curve at that score : stats.norm.pdf

x = stats.norm.rvs(loc=0, scale=1, size=1000)
plt.hist(x)
plt.show()

stats.probplot(x, plot=plt);

df = pd.read_table('./data/Receivers_2010.txt')
plt.hist(df.YdsG.values)
plt.show()

stats.probplot(df.YdsG.values, plot=plt);

