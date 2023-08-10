get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
import warnings; warnings.simplefilter('ignore')

from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns

from ipywidgets import interactive, widgets, RadioButtons, ToggleButton, Select, FloatSlider, FloatRangeSlider, IntSlider, fixed

pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
np.set_printoptions(precision=5, suppress=True) # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster')

air = pd.read_csv('data/international-airline-passengers.csv', header=0, index_col=0, parse_dates=[0])

air.head(2)

fig, ax = plt.subplots(figsize=(8,6))
air['n_pass_thousands'].plot(ax=ax)

ax.set_title('International airline passengers, 1949-1960')
ax.set_ylabel('Thousands of passengers')
ax.set_xlabel('Year')
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()

fig, ax = plt.subplots(figsize=(8,6))

air['n_pass_thousands'].resample('AS').sum().plot(ax=ax)

fig.suptitle('Aggregated annual series: International airline passengers, 1949-1960');
ax.set_ylabel('Thousands of passengers');
ax.set_xlabel('Year');
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout();
fig.subplots_adjust(top=0.9)

air['Month'] = air.index.strftime('%b')
air['Year'] = air.index.year

air.head()

# Pivot: Reshape data based on column values. pandas.DataFrame.pivot
air_piv = air.pivot(index='Year',columns='Month',values='n_pass_thousands')

air_piv.head(5)

air = air.drop(['Month','Year'], axis=1)

# Put the months in order
month_names = pd.date_range(start='2016-01-01', periods=12, freq='MS').strftime('%b')
air_piv = air_piv.reindex(columns=month_names)

fig, ax = plt.subplots(figsize=(8,6))
air_piv.plot(ax=ax, kind='box')
ax.set_xlabel('Month')
ax.set_ylabel('Thousands of passengers')
ax.set_title('Boxplot of seasonal values')
ax.xaxis.set_ticks_position('bottom')
fig.tight_layout()



