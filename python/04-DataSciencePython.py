# You can import the full scipy package, typically shortened to 'sp'
import scipy as sp

# However, it is perhaps more common to import particular submodules
#  For example, let's import the stats submodule
import scipy.stats as sts

# Let's model a fair coin - with 0.5 probability of being Heads
sts.bernoulli.rvs(0.5)

# Let's flip a bunch of coins!
coin_flips = [sts.bernoulli.rvs(0.5) for i in range(100)]
print('The first ten coin flips are: ', coin_flips[:10])
print('The percent of heads from this sample is: ', sum(coin_flips) / len(coin_flips) * 100, '%')

# Numpy is standardly imported as 'np'
import numpy as np

# Numpy's specialty is linear algebra and arrays of (uniform) data 

# Define some arrays
#  Arrays can have different types, but all the data within an array needs to be the same type
arr_1 = np.array([1, 2, 3])
arr_2 = np.array([4, 5, 6])
bool_arr = np.array([True, False, True])
str_arr = np.array(['a', 'b', 'c'])

# Note that if you try to make a mixed-data-type array, numpy won't fail, 
#  but it will (silently)
arr = np.array([1, 'b', True])

# Check the type of array items
print(type(arr[0]))
print(type(arr[2]))

# These array will therefore not act like you might expect
# The last item looks like a Boolen
print(arr[2])

# However, since it's actually a string, it won't evaluate like a Boolean
print(arr[2] == True)

# Pandas is standardly imported as pd
import pandas as pd

# Let's start with an array of data, but we also have a label for each data item
dat_1 = np.array(['London', 'Washington', 'London', 'Budapest'])
labels = ['Ada', 'Alonzo', 'Alan', 'John']

# Pandas offers the 'Series' data object to store 1d data with axis labels
get_ipython().magic('pinfo pd.Series')

# Let's make a Series with out data, and check it out
ser_1 = pd.Series(dat_1, labels)
ser_1.head()

# If we have some different data (with the same labels) we can make another Series
dat_2 = [36, 92, 41, 53]
ser_2 = pd.Series(dat_2, labels)

ser_2.head()

# However, having a collection of series can quickly get quite messy
#  Pandas therefore offer the dataframe - a powerful data object to store mixed type data with labels
get_ipython().magic('pinfo pd.DataFrame')

# There are several ways to initialize a dataframe
#  Here, we provide a dictionary made up of our series
df = pd.DataFrame(data={'Col-A': ser_1, 'Col-B':ser_2}, index=labels)

# For categorical data, we can check how many of each value there are
df['Col-A'].value_counts()

# Note that dataframes are actually collections of Series
#  When we index the df, as above, we actually pull out a Series
#    So, the '.value_counts()' is actually a Series method
type(df['Col-A'])

# Pandas also gives us tons an ways to directly explore and analyze data in dataframes
#  For example, the mean for all numberic data columns
df.mean()

# This magic command is used to plot all figures inline in the notebook
get_ipython().magic('matplotlib inline')

# Matplotlib is standardly imported as plt
import matplotlib.pyplot as plt

# Plot a basic line graph
plt.plot([1, 2, 3], [4, 6, 8])

# Import sklearn
import sklearn as skl

# Check out module description
get_ipython().magic('pinfo skl')

