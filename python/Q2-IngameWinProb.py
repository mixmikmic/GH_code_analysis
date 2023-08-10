import os
import sys
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

import statsmodels.api as sm

pandas.set_option('display.notebook_repr_html', True)
pandas.set_option('display.max_columns', 40)
pandas.set_option('display.max_rows', 25)
pandas.set_option('precision', 4)

dm = pandas.read_csv('NHLSeasonGameGoals.csv')
dm.head()

