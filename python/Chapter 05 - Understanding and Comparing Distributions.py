# %load ./snippets/data-imports
import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

## NOTE: not using seaborn here because there appears to be a bug in 
##       its handling of outliers

# http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
s = Series(np.random.normal(100, 10, 200))
s = s.append(Series([60, 65, 130, 150]))  # add a couple outliers

# Create the boxplot
plt.boxplot(s.values);
plt.xticks([1], ['group1'])
plt.xlabel("Grouping")
plt.ylabel("Values")
plt.title("Demo individual boxplot");

# matplotlib side-by-side histograms
s1 = Series(np.random.normal(100, 10, 200))
s2 = Series(np.random.normal(90, 15, 200))

# make the x axes consistent base on min/max across both groups
xlims = [min(s1.min(), s2.min()), max(s1.max(), s2.max())]
ymax = 50
bins = 15

# hist for group 1
plt.subplot(1,2,1)
plt.hist(s1, bins=bins, range=xlims) # explicitly setting bins & range keep displays consistent
plt.ylim([0, ymax])
plt.xlim(xlims)
plt.xlabel("group1")
plt.ylabel("count")

# hist for group 2
plt.subplot(1,2,2)
plt.hist(s2, bins=bins, range=xlims)
plt.ylim([0, ymax])
plt.xlim(xlims)
plt.xlabel("group2")

plt.suptitle("Side-by-Side Histogram Demo");

## create data
np.random.seed(10)
df = DataFrame(data = {
        "group1": Series(np.random.normal(100, 10, 200)),
        "group2": Series(np.random.normal(80, 30, 200)),
        "group3": Series(np.random.normal(90, 20, 200)),
        "group4": Series(np.random.normal(70, 25, 200))
    })

# Create the boxplot
plt.boxplot(df.values)
plt.xticks(range(1, len(df.columns) + 1), df.columns.values)
plt.xlabel("Groupings")
plt.ylabel("Values")
plt.title("Demo box plot of grouping values");

# using built-in pands support
df.boxplot();

