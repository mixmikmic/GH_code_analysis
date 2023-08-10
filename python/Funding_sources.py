import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

df = pd.read_csv('assets/study_fields.csv')

funded_bys_copy = df['Funded Bys']

df.replace(to_replace=['Other|NIH', 'NIH|Other'], value='Academic & NIH', inplace=True)
df.replace(to_replace=['Other|Industry', 'Industry|Other'], value='Academic & Industry', inplace=True)
df.replace(to_replace=['Industry|NIH', 'NIH|Industry'], value='Industry, NIH', inplace=True)
df.replace(to_replace=['Other|NIH|Industry', 'Other|Industry|NIH'], value='Academic, NIH, Industry', inplace=True)
df.replace(to_replace=['Other'], value='Academic', inplace=True)

# delete columns with NaNs
del df['Interventions']
del df['Outcome Measures']
del df['Study Designs']

df.dropna(axis=0, inplace=True)

interventional = df.loc[df['Study Types'] == 'Interventional']

interventional.Phases.value_counts()

dummies = pd.get_dummies(interventional['Funded Bys'], prefix_sep='')

result = pd.concat([interventional, dummies], axis=1)

phase_funders = result.groupby('Phases').sum()

categories_of_interest = ['NIH', 'Academic & NIH',
                          'Academic & Industry', 'Industry', 'Academic']

big_phase_funders = phase_funders[categories_of_interest]

big_phase_funders.plot(figsize=(10,5))

big_phase_funders

ax = sns.countplot(y='Phases', 
                   data=df, 
                   order=['Phase 0', 'Phase 1', 'Phase 1|Phase 2', 'Phase 2', 'Phase 2|Phase 3', 'Phase 3', 'Phase 4'],)



