import pandas as pd
import numpy as np

# life expectancy at birth in years
life = pd.read_csv("LINK-TO-DATA",usecols=['Country or Area','Year','Value'])
life.columns = ['country','year','life']
life[0:5]

# population
population = pd.read_csv("LINK-TO-DATA",usecols=['Country or Area', 'Year','Value'])
population.columns = ['country', 'year','population']

print "Nr of countries in life:", np.size(np.unique(life['country']))
print "Nr of countries in population:", np.size(np.unique(population['country']))

df = pd.merge(life, population, how='outer', sort=True, on=['country','year'])
df[400:405]

# Population below national poverty line, total, percentage
poverty = pd.read_csv("LINK-TO-DATA",usecols=['Country or Area', 'Year','Value'])
poverty.columns = ['country', 'year','poverty']
df = pd.merge(df, poverty, how='outer', sort=True, on=['country','year'])

# Primary school completion rate % of relevant age group by country
school = pd.read_csv("LINK-TO-DATA",usecols=['Country or Area', 'Year','Value'])
school.columns = ['country', 'year','school']
df = pd.merge(df, school, how='outer', sort=True, on=['country','year'])

# Total employment, by economic activity (Thousands)
employmentin = pd.read_csv("LINK-TO-DATA",usecols=['Country or Area', 'Year','Value','Sex','Subclassification'])
employment = employmentin.loc[(employmentin.Sex=='Total men and women')&
                              (employmentin.Subclassification=='Total.')]
employment = employment.drop('Sex', 1)
employment = employment.drop('Subclassification', 1)
employment.columns = ['country', 'year','employment']
df = pd.merge(df, employment, how='outer', sort=True, on=['country','year'])

# Births attended by skilled health staff (% of total) by country
births = pd.read_csv("LINK-TO-DATA",usecols=['Country or Area', 'Year','Value'])
births.columns = ['country', 'year','births']
df = pd.merge(df, births, how='outer', sort=True, on=['country','year'])

# Measles immunization % children 12-23 months by country
measles = pd.read_csv("LINK-TO-DATA",usecols=['Country or Area', 'Year','Value'])
measles.columns = ['country', 'year','measles']
df = pd.merge(df, measles, how='outer', sort=True, on=['country','year'])

df[0:50]

df=df.drop(df.index[0:40])

df[0:10]

df2 = df.set_index(['country','year'])

df2[0:10]

df2.describe()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.rcParams['font.size']=11
plt.rcParams['figure.figsize']=[8.0, 3.5]
fig, axes=plt.subplots(nrows=1, ncols=2)
df2.plot(kind='scatter', x='life', y='population', ax=axes[0], color='Blue');
df2.plot(kind='scatter', x='life', y='school', ax=axes[1], color='Red');
plt.tight_layout()

from pandas.tools.plotting import scatter_matrix

# group by country
grouped = df2.groupby(level=0)
dfgroup = grouped.mean()

# employment in % of total population
dfgroup['employment']=(dfgroup['employment']*1000.)/dfgroup['population']*100
dfgroup=dfgroup.drop('population',1)

scatter_matrix(dfgroup,figsize=(12, 12), diagonal='kde')

