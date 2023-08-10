# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7

# pd is a common shorthand for pandas
import pandas as pd

baby = pd.read_csv('babynames.csv')
baby

ls

baby

baby.loc[1, 'Name'] # Row labeled 1, Column labeled 'Name'

# Get rows 1 through 5, columns Name through Count inclusive
baby.loc[1:5, 'Name':'Count']

baby.loc[:, 'Year']

baby.loc[:, 'Year'] * 2

# This is a DataFrame again
baby.loc[:, ['Name', 'Year']]

# Shorthand for baby.loc[:, 'Name']
baby['Name']

# Shorthand for baby.loc[:, ['Name', 'Count']]
baby[['Name', 'Count']]

# Series of years
baby['Year']

# Compare each year with 2016
baby['Year'] == 2016

# We are slicing rows, so the boolean Series goes in the first
# argument to .loc
baby_2016 = baby.loc[baby['Year'] == 2016, :]
baby_2016

sorted_2016 = baby_2016.sort_values('Count', ascending=False)
sorted_2016

# Get the value in the zeroth row, zeroth column
sorted_2016.iloc[0, 0]

# Get the first five rows
sorted_2016.iloc[0:5]

