import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

# Read in data
trials = pd.read_csv('assets/study_fields.csv')

# Convert to numeric
trials['Enrollment'] = pd.to_numeric(trials.Enrollment, errors='coerce')

trials.Enrollment.isnull().sum()

# Create a new series without the nans
without_nulls_enrollment = trials[['NCT Number', 'Enrollment', 'Study Types', 'Recruitment']].dropna()

# alright, there's clear outlier
without_nulls_enrollment['Enrollment'].max()

# View all the trials with enrollment of greater than 20k
without_nulls_enrollment[without_nulls_enrollment['Enrollment'] > 20000].shape

# How many interventional trials with greater than 100k participants?
# There are still a good number of trials with over 100k participants, so not comfortable removing outliers
# I would need to remove outliers if I wanted to do a swarm plot
without_nulls_enrollment[without_nulls_enrollment['Enrollment'] > 100000]['Study Types'].value_counts()

# View enrollment stats based on each recruitment category
without_nulls_enrollment.groupby('Recruitment').agg([np.mean, np.count_nonzero,])

# Remove the trials with recruitment status withdrawn and terminated
without_nulls_enrollment = without_nulls_enrollment.loc[
    (without_nulls_enrollment['Recruitment'] != 'Withdrawn') & (without_nulls_enrollment['Recruitment'] != 'Terminated')]

# bin data
bins = [-1, 30, 60, 100, 200, 400, 1000, 5000, 1000000000]
group_names = ['< 30', '31-60', '61-100', '101-200', '201-400', '401-1000', '1000-5000', '>5000']
categories = pd.cut(without_nulls_enrollment['Enrollment'], bins, labels=group_names)

# Add categories as column in dataframe
without_nulls_enrollment['Category'] = categories

# View value counts
enrollment_counts = without_nulls_enrollment['Category'].value_counts().sort_index(ascending=True)

# plot
enrollment_counts.plot(kind='bar', title='Size of Cancer Trials', alpha=0.6, colormap='Accent', rot=20)

# View value counts for Interventional vs. observational
without_nulls_enrollment['Study Types'].value_counts()

# Create dummies for study type - this will enable a stacked bar graph
dummies = pd.get_dummies(without_nulls_enrollment['Study Types'], prefix_sep='')

# Join dummies with df
result = pd.concat([without_nulls_enrollment, dummies], axis=1)

# Create a groupby object that can be converted to a stacked bar graph
enrollment_by_study_type =  result.groupby('Category').sum()[['Interventional', 'Observational']]

# Sort the groupby object you just created
groupby_sorted = enrollment_by_study_type.sort_index(ascending=False)

groupby_sorted.plot(kind='barh', stacked=True, alpha=0.6, colormap='Accent')

# build out list of tuples with enrollment category and amount of total trials in that category
tuples = []
for cat in without_nulls_enrollment['Category'].unique().sort_values():
    tuples.append((without_nulls_enrollment.loc[without_nulls_enrollment['Category'] == cat].shape[0], cat))

tuples

# build out another list of tuples, this time with the number of interventional trials per category
int_per_cat = []
for total, cat in tuples:
    int_per_cat.append((without_nulls_enrollment.loc[(without_nulls_enrollment['Study Types'] == 'Interventional') & (without_nulls_enrollment['Category'] == cat)].shape[0], cat))

int_per_cat

# zip together the totals and the interventionals
tot_vs_int = zip([i[0] for i in tuples], [i[0] for i in int_per_cat])

tot_vs_int

# now just get a list of the percent of interventional trials per category
percent_int_per_cat = []
for total, inter in tot_vs_int:
    percent_int_per_cat.append(float(inter)/total)

percent_int_per_cat

# now i need to repeat for observational
int_per_obs = []
for total, cat in tuples:
    int_per_obs.append((without_nulls_enrollment.loc[(without_nulls_enrollment['Study Types'] == 'Observational') & (without_nulls_enrollment['Category'] == cat)].shape[0], cat))
int_per_obs

# now zip together with totals
tot_vs_obs = zip([i[0] for i in tuples], [i[0] for i in int_per_obs])
tot_vs_obs

# get percentages
percent_int_per_obs = []
for tot, obs in tot_vs_obs:
    percent_int_per_obs.append(float(obs)/tot)
percent_int_per_obs

# create df for the percent interventionals vs categories
categories = [i[1] for i in tuples]
int_df = pd.DataFrame({'category':categories, 'percent': percent_int_per_cat, 'study-type':'interventional'})

int_df

# create df for percent observationals vs categories
obs_df = pd.DataFrame({'category': categories, 'percent': percent_int_per_obs, 'study-type':'observational'})

obs_df

# concat int_df and obs_df
frames = [int_df, obs_df]
df_for_plot = pd.concat(frames, ignore_index=True)

# multiply percent column by 100
df_for_plot['percent'] = df_for_plot['percent'] * 100

sns.pointplot(x='category', y='percent', hue='study-type', data=df_for_plot)

df_for_plot



