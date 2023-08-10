get_ipython().magic('matplotlib inline')
from IPython.core.pylabtools import figsize
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')
figsize(11,9)

import scipy.stats as stats

import pymc as pm

df = pd.read_csv('data/all_years_mass_shootings.csv', parse_dates=['Incident Date'])
df.head()

df.Operations.describe()

df = df.drop('Operations', 1)
df.head()

df.State.describe()

state_groups = df.groupby('State')
state_counts = state_groups.State.count()
state_counts = state_counts.sort_values(ascending=False)
state_counts.plot(kind='bar')

state_counts.describe()

def count_affected(x): 
    return x['# Killed'].sum() + x['# Injured'].sum()

killed_counts = state_groups['# Killed'].sum()
injured_counts = state_groups['# Injured'].sum()
affected_counts = state_groups.apply(count_affected)
impact = pd.DataFrame(data={
        "killed": killed_counts,
        "injured": injured_counts,
        "affected": affected_counts})
impact = impact.sort_values(by=['affected'], ascending=False)
impact[['killed', 'injured']].plot(kind='bar', stacked=True)

demographic_filename = "data/zip_code_demographics.csv"
demographics = pd.read_csv(demographic_filename)
state_groups = demographics.groupby('State')
state_populations = state_groups['EstimatedPopulation'].sum()
state_populations.dropna(inplace=True)
state_populations = pd.DataFrame(data=state_populations)
state_populations.head()

state_table = pd.read_csv('data/state_table.csv')
state_table.head()

populated_states = pd.merge(state_populations, state_table[['abbreviation', 'name']], left_index="State", right_on='abbreviation')
populated_states.head()

def per_hundred_k(row):
    return(row.affected / row.EstimatedPopulation) * 100000

per_capita_impact = pd.merge(populated_states, impact, left_on='name', right_index='State', how='left')
per_capita_impact.fillna(0, inplace=True)
per_capita_impact['per_hundred_k'] = per_capita_impact.apply(per_hundred_k, axis=1)
per_capita_impact = per_capita_impact.sort_values(by=['per_hundred_k'], ascending=False)
per_capita_impact.head()

per_capita_impact.per_hundred_k.describe()

# df.plot(kind='bar', x=per_capita_impact.abbreviation, y=per_capita_impact.per_hundred_k)
x = per_capita_impact.per_hundred_k.plot(kind='bar')

g = per_capita_impact.groupby('name')
df = pd.DataFrame(data=g.per_hundred_k.mean(), index=per_capita_impact.name)
df.plot(kind='bar')
df.head()



