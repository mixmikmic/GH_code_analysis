import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Increase figure and font sizes
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

countries = pd.read_csv('https://github.com/estimand/teaching-datasets/raw/master/world-development-indicators/wdi-countries.csv.gz')
wdi = pd.read_csv('https://github.com/estimand/teaching-datasets/raw/master/world-development-indicators/wdi-data.csv.gz')

indicators = {
    'NY.GNP.PCAP.CD': 'gni_per_capita',  # GNI per capita, Atlas method (current US$)
    'SP.DYN.IMRT.IN': 'infant_mort',     # Mortality rate, infant (per 1,000 live births)
    'SP.DYN.LE00.IN': 'life_exp'         # Life expectancy at birth, total (years)
}

wdi_subset = wdi[(wdi['indicator_code'].isin(indicators.keys())) & (wdi['year'] == 2015)].copy()
wdi_subset.drop('year', axis=1, inplace=True)
wdi_subset['indicator_code'].replace(indicators, inplace=True)

wdi_subset.head()

wdi_subset = wdi_subset.pivot(index='country_code', columns='indicator_code', values='value')
wdi_subset.dropna(inplace=True)

wdi_subset.head()

wdi_subset = pd.merge(wdi_subset, countries, left_index=True, right_on='country_code')

wdi_subset.head()

wdi_subset['life_exp'].plot.hist()

# Try changing the number of bins
wdi_subset['life_exp'].plot.hist(20)

# Add title and labels
wdi_subset['life_exp'].plot.hist(20, title='Histogram of life expectancy')
plt.xlabel('Life expectancy (years)')
plt.ylabel('Frequency')

# Compare with the corresponding density plot
wdi_subset['life_exp'].plot.density()

# Grouped histograms
wdi_subset.hist(column='life_exp', by='income_group')

# Grouped histograms with shared x-axis
wdi_subset.hist(column='life_exp', by='income_group', sharex=True)

# Grouped histograms with shared x- and y-axes
wdi_subset.hist(column='life_exp', by='income_group', sharex=True, sharey=True)

wdi_subset.plot.scatter(x='infant_mort', y='life_exp')

# Add transparency
wdi_subset.plot.scatter(x='infant_mort', y='life_exp', alpha=0.3)

# Vary point colour by GNI per capita
wdi_subset.plot.scatter(x='infant_mort', y='life_exp', c='gni_per_capita', colormap='Blues')

# Scatter matrix of three (numerical) variables
pd.plotting.scatter_matrix(wdi_subset[['infant_mort', 'life_exp', 'gni_per_capita']], figsize=(10, 8))

# Compute average life expectancy and infant mortality for each country group
wdi_subset.groupby('income_group')[['life_exp', 'infant_mort']].mean()

# Plot side-by-side
wdi_subset.groupby('income_group')[['life_exp', 'infant_mort']].mean().plot.bar()

# Five-number summary (min, Q1, Q2 [median], Q3, max)
wdi_subset['life_exp'].describe()

# Compare with box plot
wdi_subset['life_exp'].plot.box()

# Include multiple variables
wdi_subset[['life_exp', 'infant_mort']].plot.box()

# Grouped box plots
wdi_subset.boxplot('life_exp', by='income_group')

uk_life_exp = wdi[(wdi['country_code'] == 'GBR') & (wdi['indicator_code'] == 'SP.DYN.LE00.IN')]

uk_life_exp.head()

uk_life_exp.plot.line(x='year', y='value')

