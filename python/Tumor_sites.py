import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

# Read in data
trials = pd.read_csv('assets/study_fields.csv', encoding='utf-8')

# List of cancer sites
cancer_sites = [('breast', 'Breast'), ('lung', 'Lung'), ('colo', 'Colorectal'), ('prostate', 'Prostate'),
                ('pancrea', 'Pancreatic'), ('thyroid', 'Thyroid'), ('ovar', 'Ovarian'), ('melanoma', 'Melanoma'),
               ('esoph', 'Esophageal'), ('myeloma', 'Multiple Myeloma'), ('lymphoma', 'Lymphomas'),
                ('leukemia', 'Leukemias'), ('uter', 'Uterine'), ('bladder', 'Bladder'), ('cerv', 'Cervical'),
               ('head and neck', 'Head and Neck'), ('liver', 'Liver'), ('testi', 'Testicular')]

# Add columns for cancer sites
for search_term, site in cancer_sites:
    trials[site] = trials.Conditions.str.contains(search_term, case=False)
    trials[site] = trials[site].map({True:1, False:0})

# List of cancer sites with multiple names
cancer_sites_mult_names = [(('brain', 'glio'), 'Brain'), (('kidney', 'renal'), 'Kidney'), (('stomach', 'gastric'), 'Gastric'),
                           (('bone', 'osteos'), 'Bone'), (('soft tissue', 'gastrointestinal stromal'), 'Soft-tissue')]

# Add additional columns for cancer sites with multiple search terms
for search_terms, site in cancer_sites_mult_names:
    trials[site] = ((trials.Conditions.str.contains(search_terms[0], case=False)) | 
                     (trials.Conditions.str.contains(search_terms[1], case=False)))
    trials[site] = trials[site].map({True:1, False:0})

# Number of cancer sites - for number of bars on plot
num_cancer_sites = np.arange(len(cancer_sites) + len(cancer_sites_mult_names))

# Trial totals - for length of bars
trial_totals_by_site = trials.iloc[:, -23:].sum().sort_values(ascending=False).values

# Names of cancer sites - for bar labels
cancer_sites_high_to_low = trials.iloc[:, -23:].sum().sort_values(ascending=False).index

# Create horizontal bar
plt.barh(num_cancer_sites, trial_totals_by_site, align='center', alpha=0.4)

# Create yticks
plt.yticks(num_cancer_sites, cancer_sites_high_to_low)

# Create xlabel
plt.xlabel('Number of Trials')

# Invert graph
plt.gca().invert_yaxis()

# graph with plotly

import plotly.plotly as py
import plotly.graph_objs as go

# x is name of the condition, y is the number of trials for that condition
x = trials.iloc[:, -23:].sum().sort_values(ascending=False).index.values
y = trials.iloc[:, -23:].sum().sort_values(ascending=False).values

data = [
    go.Bar(
    x = trials.iloc[:, -23:].sum().sort_values(ascending=True).values,
    y = trials.iloc[:, -23:].sum().sort_values(ascending=True).index.values,
    orientation='h')
]

layout = go.Layout(margin=dict(
    l=120, r=30, b=60, t=60))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='cancer_counts_hbar')

trials.columns

# split pipe delimited sponsor names into a list in each cell
s = trials['Sponsor/Collaborators'].str.split('|')

# The lead sponsor is the first one listed - generate new list with only lead sponsor
lead_sponsors = [row[0] for row in s]

# Turn lead_sponsors list to a pandas series
lead_sponsors_series = pd.Series(lead_sponsors)

trials['lead_sponsors_series'] = lead_sponsors_series

top_hundred = trials.lead_sponsors_series.value_counts().sort_values(ascending=False)[:200].index

# here i'm trying to get all the sponsor and collaborators. notice how NCI shoots up
sponsors = []
tot_trials_sponsored = []
for spons in top_hundred:
    x = 0
    for row in s:
        for i in row:
            if i == spons:
                x +=1
    sponsors.append(spons)
    tot_trials_sponsored.append(x)
sponsors_dict = dict(sponsor=sponsors, trial_count=tot_trials_sponsored)

tot_trials_sponsored_df = pd.DataFrame(sponsors_dict)

tot_trials_sponsored_df.sort_values(by='trial_count', ascending=False, inplace=True)

tot_trials_sponsored_df.set_index(keys='sponsor', drop=False, inplace=True)

lead_sponsors_df = pd.DataFrame(trials['lead_sponsors_series'].value_counts().sort_values(ascending=False)[:200])

result = pd.concat([lead_sponsors_df, tot_trials_sponsored_df], axis=1)

result.sort_values(by='trial_count', inplace=True, ascending=True)

result['collaborator'] = result.trial_count - result.lead_sponsors_series

result.rename(columns={'sponsor': 'Sponsor/Collaborators'}, inplace=True)

result_two = pd.merge(result, trials, how='inner', on='Sponsor/Collaborators')

trials.drop('Rank', axis=1, inplace=True)

total_captured = pd.DataFrame(trials.groupby('lead_sponsors_series').sum().sum(axis=1).sort_values(ascending=False))

total_true = pd.DataFrame(trials['lead_sponsors_series'].value_counts())

combined = pd.merge(total_captured, total_true, left_index=True, right_index=True)

top_forty_two = tot_trials_sponsored_df.iloc[:42,:]

final_df = pd.DataFrame(trials.groupby('lead_sponsors_series').sum().ix[top_forty_two.index, :])

top_sites_per_spons = pd.DataFrame(final_df.idxmax(axis=1))

top_sites_per_spons['num_trials'] = final_df.max(axis=1).values

top_sites_per_spons

trials.groupby('lead_sponsors_series').sum().sort_values(by='Breast', ascending=False).iloc[:10, :].sum(axis=1)

trials.lead_sponsors_series.value_counts().sort_values(ascending=False)

# left off - need to create an 'other' category for all the cancer types not captured



