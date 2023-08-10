import pandas as pd                 # Our pandas package
from pandas_datareader import data  # This is will give us access to FRED
import datetime as dt               # This will get us the datatime package
import matplotlib.pyplot as plt     # The new package we are learning about Matplotlib
                                    # pyplot is just one module of many in this library
    
get_ipython().magic('matplotlib inline')

url = "http://pages.stern.nyu.edu/~dbackus/Data/debt.csv"

debt = pd.read_csv(url)

type(debt)

debt.head() # This one is not on the list but gives us a quick peak at it

debt.shape

debt.columns

debt.index

debt.dtypes

debt.set_index("Year", inplace = True)

debt.head() # Here we see that the index is the year

# HEre is the way to do it via a dictionary

country_dict = {"ARG": "Argentina", "DEU": "Germany", "GRC": "Greece"}

debt.rename(columns = country_dict).head(10)

# Note that this is not set, so if we want to keep this change, we need to do
# inplace  = true

# then this is the way to to it with a list, 
# here I'm converting the dictionary values to a list

debt.columns = list(country_dict.values())

debt.head()

debt.mean()

debt.mean(axis = 1)

debt.mean().mean()

# This is a way to comput the mean across all things...

debt.plot(linewidth = 2)

fig, ax = plt.subplots()

my_colors = ['red', 'green', 'blue']

debt.plot(ax = ax, linewidth = 2, color = my_colors)
debt["Argentina"].plot(ax = ax, linewidth = 6, color = "red")

ax.set_ylim(0,200)
ax.set_title("Debt")
ax.set_ylabel("Public Debt (% of GDP)")



