# Practice: Pythagorean winning percentage

get_ipython().magic('matplotlib inline')
import os
import sys
import pandas
import numpy
import matplotlib
import matplotlib.pyplot as plt
pandas.set_option('display.notebook_repr_html', True)
pandas.set_option('display.max_columns', 20)
pandas.set_option('display.max_rows', 25)
pandas.set_option('precision',3)

from decimal import getcontext, Decimal
# Set the precision.
getcontext().prec = 3

dm = pandas.read_csv('NHL_season_team_goals.csv')
dm = dm[['Team', 'GamesPlayed', 'Wins', 'Losses', 'GoalsFor', 'GoalsAllowed']]
dm.head()



