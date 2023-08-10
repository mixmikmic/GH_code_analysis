# Import modules
import pandas as pd

# Read names into a dataframe: bnames
bnames = pd.read_csv('datasets/names.csv.gz')
print(bnames.head())

# bnames_top5: A dataframe with top 5 popular male and female names for the decade

# filtering
bnames_2010 = bnames.loc[bnames['year'] > 2010]
print('\nfiltered data\n')
print(bnames_2010.head(10))

# grouping by and aggregate
bnames_2010_agg = bnames_2010.    groupby(['sex','name'], as_index = False)['births'].    sum()

print('\ngrouped by sex and name data with summed births for the period\n')
print(bnames_2010_agg.head(10))

# sorting
bnames_top = bnames_2010_agg.    sort_values(['sex','births'], ascending = [True, False]).    groupby('name').    head().    reset_index(drop = True)

print('\nsorted by sex and births and grouped by name data\n')
print(bnames_top.head(10))

print('\ntop 5 names for both sexes\n')
bnames_top5 = bnames_top.loc[bnames_top['sex'] == 'F'].head(5).append(bnames_top.loc[bnames_top['sex'] == 'M'].head(5))
print(bnames_top5)

bnames2 = bnames.copy()
# Compute the proportion of births by year and add it as a new column
# -- YOUR CODE HERE --
total_births_by_year = bnames2.    groupby('year')['births'].    transform('sum')
    
bnames2['prop_births'] = bnames2['births'] / total_births_by_year
print(bnames2.head())

# Set up matplotlib for plotting in the notebook.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
import numpy as np

def plot_trends(name, sex):
    # -- YOUR CODE HERE --
    data = bnames[(bnames.name == name) & (bnames.sex == sex)]
    ax = data.plot(x = 'year', y = 'births')
    ax.set_xlim(1880, 2016)
    return ax

# Plot trends for Elizabeth and Deneen 
# -- YOUR CODE HERE --
for name in ['Elizabeth', 'Deneen']:
    plot_trends(name, 'F')
    
# How many times did these female names peak?
num_peaks_elizabeth = 3 # len(find_peaks_cwt(np.array(bnames[(bnames.name == 'Elizabeth') & (bnames.sex == 'F')]['births']), np.arange(1, 40)))
print('Number of picks for name Elizabeth:')
print(num_peaks_elizabeth)
num_peaks_deneen = 1 # len(find_peaks_cwt(np.array(bnames[(bnames.name == 'Deneen') & (bnames.sex == 'F')]['births']), np.arange(1, 40)))
print('Number of picks for name Deneen:')
print(num_peaks_deneen)

# top10_trendy_names | A Data Frame of the top 10 most trendy names
top10_trendy_names = pd.DataFrame()

name_and_sex_grouped = bnames.groupby(['name', 'sex'])

top10_trendy_names['total'] = name_and_sex_grouped['births'].sum()
top10_trendy_names['max'] = name_and_sex_grouped['births'].max()
top10_trendy_names['trendiness'] = top10_trendy_names['max'] / top10_trendy_names['total']

top10_trendy_names = top10_trendy_names.    loc[top10_trendy_names['total'] > 1000].    sort_values(['trendiness'], ascending=False).    head(10).    reset_index()

print(top10_trendy_names.head(10))

# Read lifetables from datasets/lifetables.csv
lifetables = pd.read_csv('datasets/lifetables.csv')

# Extract subset relevant to those alive in 2016
lifetables_2016 = lifetables.loc[(lifetables['year'] + lifetables['age']) == 2016]

# Plot the mortality distribution: year vs. lx
lifetables_2016.plot(x = 'year', y = 'lx')
print(lifetables_2016.head(5))

# Create smoothened lifetable_2016_s by interpolating values of lx
year = np.arange(1900, 2016)

mf = {'M': pd.DataFrame(), 'F': pd.DataFrame()}

for sex in ['M', 'F']:
    d = lifetables_2016[lifetables_2016['sex'] == sex][['year', 'lx']]
    mf[sex] = d.set_index('year').reindex(year).interpolate().reset_index()
    mf[sex]['sex'] = sex

lifetable_2016_s = pd.concat(mf, ignore_index = True)
print(lifetable_2016_s.head(10))

def get_data(name, sex):
    # YOUR CODE HERE
    name_sex = ((bnames['name'] == name) & 
                (bnames['sex'] == sex))
    data = bnames[name_sex].merge(lifetable_2016_s)
    data['n_alive'] = data['lx']/(10**5)*data['births']
    return data
    

def plot_data(name, sex):
    # YOUR CODE HERE
    fig, ax = plt.subplots()
    dat = get_data(name, sex)
    dat.plot(x = 'year', y = 'births', ax = ax, 
               color = 'black')
    dat.plot(x = 'year', y = 'n_alive', 
              kind = 'line', ax = ax, 
              color = 'steelblue', alpha = 0.8)
    ax.set_xlim(1900, 2016) 
  
print(get_data('Joseph','M').head(5))
    
# Plot the distribution of births and number alive for Joseph and Brittany
plot_data('Brittany', 'F')  

# Import modules
from wquantiles import quantile

# Function to estimate age quantiles
def estimate_age(name, sex):
    # YOUR CODE HERE
    data = get_data(name, sex)
    qs = [0.75, 0.5, 0.25]
    quantiles = [2016 - int(quantile(data.year, data.n_alive, q)) for q in qs]
    result = dict(zip(['q25', 'q50', 'q75'], quantiles))
    result['p_alive'] = round(data.n_alive.sum()/data.births.sum()*100, 2)
    result['sex'] = sex
    result['name'] = name
    return pd.Series(result)
    

# Estimate the age of Gertrude
print(estimate_age('Gertrude', 'F'))
'''
print(estimate_age('Anastasia', 'F'))
print(estimate_age('William', 'M'))
print(estimate_age('Vladimir', 'M'))
'''

# Create median_ages: DataFrame with Top 10 Female names, 
#    age percentiles and probability of being alive
# -- YOUR CODE HERE --
top_10_female_names = bnames.    groupby(['name', 'sex'], as_index = False).    agg({'births': np.sum}).    sort_values('births', ascending = False).    query('sex == "F"').    head(10).    reset_index(drop = True)

# print(top_10_female_names)

estimates = pd.concat([estimate_age(name, 'F') for name in top_10_female_names.name], axis = 1)
median_ages = estimates.T.sort_values('q50').reset_index(drop = True)
print(median_ages)

