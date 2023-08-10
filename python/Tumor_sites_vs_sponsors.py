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

# split pipe delimited sponsor names into a list in each cell
s = trials['Sponsor/Collaborators'].str.split('|')

# The lead sponsor is the first one listed - generate new list with only lead sponsor
lead_sponsors = [row[0] for row in s]

# Turn lead_sponsors list to a pandas series
lead_sponsors_series = pd.Series(lead_sponsors)

# create seriers from list
trials['lead_sponsors_series'] = lead_sponsors_series

# drop the rank column
trials.drop('Rank', axis=1, inplace=True)

# list of sponsors with most trials sponsored
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

# convert list to dataframe
tot_trials_sponsored_df = pd.DataFrame(sponsors_dict)

tot_trials_sponsored_df.sort_values(by='trial_count', ascending=False, inplace=True)

tot_trials_sponsored_df.set_index(keys='sponsor', drop=False, inplace=True)

# get top 42 sponsors to go along with the number in the charts on plotly
top_forty_two = tot_trials_sponsored_df.iloc[:42,:]

# this gives you the sum of each cancer type grouped by sponsor
final_df = pd.DataFrame(trials.groupby('lead_sponsors_series').sum().ix[top_forty_two.index, :])

# from this i get the index of each maximum value per row
top_sites_per_spons = pd.DataFrame(final_df.idxmax(axis=1))

# adding number of trials to that dataframe i just created
top_sites_per_spons['num_trials'] = final_df.max(axis=1).values

top_sites_per_spons



