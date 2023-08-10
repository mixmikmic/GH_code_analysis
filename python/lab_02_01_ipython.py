# To run a python block, after writing your commands, press [shift] + [enter]
# Unless you 'print' results, only the last executed statement can print to output
3+4
5+10

x = 5
y = 25

y = x + y
print y

# Imports work just as expected. Note! If you want plots to show, add the inline comment below
import matplotlib as plt
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')

# To get function help docstrings instead of autocomplete, press [shift] + [tab]
x = pd.Series(np.random.rand(100),index=pd.date_range('1/1/2015',periods=100))
x.head()

x = x.cumsum()
x.head()

x.plot()

