# data types
x = 4
type(x)

pi = 3.14
type(pi)

name = 'my string'
type(name)

# data structures / objects

my_list = [2, 4, 10]  # a list

my_list[2]  # access by position

my_dict = {'pi': 3.14, 'd': 4}  # a dictionary


my_dict['pi']  # access by key

import numpy as np

x = np.zeros(shape=(4, 5))
x

y = x + 4
y

# random numbers
z = np.random.random(x.shape)
z

# aggregations
z_sum = z.sum(axis=1)
z_sum

# broadcasting
y.transpose() * z_sum

# slicing
z[2:4, ::2]  # 2-4 on the first axis, stride of 2 on the second

# data types

xi = np.array([1, 2, 3], dtype=np.int)  # integer
xi.dtype

xf = np.array([1, 2, 3], dtype=np.float)  # float
xf.dtype

# universal functions (ufuncs, e.g. sin, cos, exp, etc)
np.sin(z_sum)

import pandas as pd

# This data can also be loaded from the statsmodels package
# import statsmodels as sm
# co2 = sm.datasets.co2.load_pandas().data 

co2 = pd.read_csv('./data/co2.csv', index_col=0, parse_dates=True)

# co2 is a pandas.DataFrame
co2.head()  # head just prints out the first few rows

# The pandas DataFrame is made up of an index
co2.index

# and 0 or more columns (in this case just 1 - co2)
# Each column is a pandas.Series
co2['co2'].head()  

# label based slicing
co2['1990-01-01': '1990-02-14']

# aggregations just like in numpy
co2.mean(axis=0)

# advanced grouping/resampling

# here we'll calculate the annual average timeseris of co2 concentraions
co2_as = co2.resample('AS').mean()  # AS is for the start of each year

co2_as.head()

# we can also quickly calculate the monthly climatology

co2_climatology = co2.groupby(co2.index.month).mean()
co2_climatology

get_ipython().run_line_magic('matplotlib', 'inline')

# and even plot that using pandas and matplotlib
co2_climatology.plot()



