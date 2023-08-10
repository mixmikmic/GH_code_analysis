import numpy as np
import pandas as pd

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# Increase figure and font sizes
sns.set(rc={
    'figure.figsize': (8, 6),
    'font.size': 14
})

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

sns.distplot(wdi_subset['life_exp'])

# Add title and labels
sns.distplot(wdi_subset['life_exp']).set(xlabel='Life expectancy (years)', ylabel='Frequency')

# Scatter plot with regression line
sns.regplot(x='infant_mort', y='life_exp', data=wdi_subset)

# Scatter plot with marginal histograms
sns.jointplot(x='infant_mort', y='life_exp', data=wdi_subset)

# 2-D density plot with marginal densities
sns.jointplot(x='infant_mort', y='life_exp', data=wdi_subset, kind='kde')

# Scatter matrix of three (numerical) variables
sns.pairplot(wdi_subset[['infant_mort', 'life_exp', 'gni_per_capita']])

# Plot average life expectancy by country group
sns.barplot(x='income_group', y='life_exp', data=wdi_subset)

# Plot median life expectancy by country group
sns.barplot(x='income_group', y='life_exp', data=wdi_subset, estimator=np.median)

# Five-number summary (min, Q1, Q2 [median], Q3, max)
wdi_subset['life_exp'].describe()

# Compare with box plot
sns.boxplot(y='life_exp', data=wdi_subset)

# Grouped box plots
sns.boxplot(x='income_group', y='life_exp', data=wdi_subset)

uk_life_exp = wdi[(wdi['country_code'] == 'GBR') & (wdi['indicator_code'] == 'SP.DYN.LE00.IN')]

uk_life_exp.head()

sns.pointplot(x='year', y='value', data=uk_life_exp, markers='')

