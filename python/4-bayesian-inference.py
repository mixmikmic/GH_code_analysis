import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
sns.set_style('whitegrid')
sns.set_context('poster')
pd.options.display.max_columns = 30

all_teams = pd.read_csv('../data/all-teams-1990-2016.csv')

all_teams.head()

import pymc3 as pm
import pydot

sat = all_teams[all_teams.weekday_name == 'Saturday'].attendance.values
wed = all_teams[all_teams.weekday_name == 'Wednesday'].attendance.values

prior = all_teams[(all_teams.weekday_name != 'Saturday') & (all_teams.weekday_name != 'Wednesday')].attendance.values

prior_mean = np.mean(prior)
prior_std = np.std(prior)

print prior_mean
print prior_std

with pm.Model() as model:

    sat_mean = pm.Normal('sat_mean', prior_mean, sd=prior_std)
    wed_mean = pm.Normal('wed_mean', prior_mean, sd=prior_std)

std_prior_lower = 5000.0
std_prior_upper = 15000.0

with model:
    
    sat_std = pm.Uniform('sat_std', lower=std_prior_lower, upper=std_prior_upper)
    wed_std = pm.Uniform('wed_std', lower=std_prior_lower, upper=std_prior_upper)

with model:

    grp_sat = pm.Normal('group_sat', mu=sat_mean, sd=sat_std, observed=sat)
    grp_wed = pm.Normal('group_wed', mu=wed_mean, sd=wed_std, observed=wed)

with model:

    diff_of_means = pm.Deterministic('difference of means', sat_mean - wed_mean)
    diff_of_stds = pm.Deterministic('difference of stds', sat_std - wed_std)
    effect_size = pm.Deterministic('effect size', diff_of_means / np.sqrt((sat_std**2 + wed_std**2) / 2))

with model:
    trace = pm.sample(20000, njobs=-1)

pm.plot_posterior(trace[3000:], varnames=['sat_mean', 'wed_mean', 'sat_std', 'wed_std'])

pm.plot_posterior(trace[3000:], varnames=['difference of means', 'difference of stds', 'effect size'], ref_val=0)

yearly = pd.read_csv('../data/yearly-team-data.csv')

yearly.head()

active_parks = yearly
for i in yearly.stadium.unique():
    if i not in yearly[yearly.year == 2016].stadium.unique():
        active_parks = active_parks[active_parks.stadium != i]

parks_summary = pd.DataFrame(columns=['stadium', 'first_year', 'avg_capacity', 'capacity_std', 'avg_win_pct', 'avg_attendance'])
for i in active_parks.stadium.unique():
    parks_summary.loc[len(parks_summary)] = [
        i,
        min(active_parks[active_parks.stadium == i].year),
        round(np.mean(active_parks[active_parks.stadium == i].capacity),0) / 1000,
        np.std(active_parks[active_parks.stadium == i].capacity),
        round(np.mean(active_parks[active_parks.stadium == i].win_pct), 3),
        round(np.mean(active_parks[active_parks.stadium == i].attendance),0) / 1000000
    ]

parks_summary.sort_values('avg_capacity').reset_index(drop=True)

bayes = all_teams[all_teams.year >= 2008]

wsn = bayes[bayes.team_x == 'WSN'].attendance.values
nym = bayes[(bayes.team_x == 'NYM') & (bayes.stadium == 'Citi Field')].attendance.values

prior = bayes[(bayes.team_x != 'WSN') & (bayes.team_x != 'NYM')].attendance.values

prior_mean = np.mean(prior)
prior_std = np.std(prior)

print prior_mean
print prior_std

with pm.Model() as model:

    wsn_mean = pm.Normal('wsn_mean', prior_mean, sd=prior_std)
    nym_mean = pm.Normal('nym_mean', prior_mean, sd=prior_std)

std_prior_lower = 5000.0
std_prior_upper = 15000.0

with model:
    
    wsn_std = pm.Uniform('wsn_std', lower=std_prior_lower, upper=std_prior_upper)
    nym_std = pm.Uniform('nym_std', lower=std_prior_lower, upper=std_prior_upper)

with model:

    grp_wsn = pm.Normal('group_wsn', mu=wsn_mean, sd=wsn_std, observed=wsn)
    grp_nym = pm.Normal('group_nym', mu=nym_mean, sd=nym_std, observed=nym)

with model:

    diff_of_means = pm.Deterministic('difference of means', wsn_mean - nym_mean)
    diff_of_stds = pm.Deterministic('difference of stds', wsn_std - nym_std)
    effect_size = pm.Deterministic('effect size', diff_of_means / np.sqrt((wsn_std**2 + nym_std**2) / 2))

with model:
    trace = pm.sample(20000, njobs=-1)

pm.plot_posterior(trace[3000:], varnames=['wsn_mean', 'nym_mean', 'wsn_std', 'nym_std'])

pm.plot_posterior(trace[3000:], varnames=['difference of means', 'difference of stds', 'effect size'], ref_val=0)



