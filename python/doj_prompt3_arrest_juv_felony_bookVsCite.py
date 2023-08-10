

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

# from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np

import re

from calendar import isleap

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
np.set_printoptions(precision=5, suppress=True) # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster')

import os
import sys
sys.path.append(os.path.join(os.getenv('HOME'),'src','python'))
import classify_utils as cu

# Load contextual indicators

# df_ctx = pd.read_csv('ca_county_agency_contextual_indicators_2009-2014_05-03-2016.csv', parse_dates=['year'])
df_ctx = pd.read_csv('ca_county_agency_contextual_indicators_2009-2014_05-03-2016.csv')
df_ctx = df_ctx.set_index('year')
# df_ctx.index = df_ctx.index.to_period(freq='A')

df_ctx.head()

df_ctx.columns

# Load county population

# df_pop = pd.read_csv('ca_county_population_by_race_gender_age_2005-2014_02-05-2016.csv', parse_dates=['year'])
df_pop = pd.read_csv('ca_county_population_by_race_gender_age_2005-2014_02-05-2016.csv')
df_pop = df_pop.set_index('year')
df_pop = df_pop.ix['2009':]
# df_pop.index = df_pop.index.to_period(freq='A')

df_pop['gender'].unique()

df_pop.head(5)

# get only the 'All Combined' rows, combining the agencies for each county
df_ctx_combo = df_ctx[df_ctx['agency_code'] == 'All Combined']

# drop these columns because they all contain the same values
df_ctx_combo = df_ctx_combo.drop(['agency_name', 'agency_code'], axis=1)
# make a multiindex
df_ctx_combo = df_ctx_combo.set_index('county', append=True)

df_ctx_combo.head()

# get per-county 'All Combined' rows of population data, to join to the contextual data
df_pop_combo = df_pop[(df_pop['county'] != 'All Combined') &
                      (df_pop['race'] == 'All Combined') &
                      (df_pop['gender'] == 'All Combined') &
                      (df_pop['age_group'] == 'All Combined')]

# drop these columns because we're using 'All Combined'
df_pop_combo = df_pop_combo.drop(['race', 'gender', 'age_group'], axis=1)
# make a multiindex so we can join them
df_pop_combo = df_pop_combo.set_index('county', append=True)

# create features for: population proportion of county by age, gender, race

# not sure whether these will be useful, just exploratory

ages = ['Adult', 'Juvenile']
for age in ages:
    df_this_pop = pd.DataFrame()
    df_this_pop = df_pop[(df_pop['county'] != 'All Combined') &
                          (df_pop['race'] == 'All Combined') &
                          (df_pop['gender'] == 'All Combined') &
                          (df_pop['age_group'] == age)].drop(['race', 'gender', 'age_group'], axis=1).set_index('county', append=True)
    col_str = 'prop_age_{}'.format(age)
    df_this_pop = df_this_pop.rename(columns={'population': col_str})
    df_this_pop[col_str] = df_this_pop[col_str] / df_pop_combo['population']
    
    # join it with the larger df
    df_pop_combo = df_pop_combo.merge(df_this_pop, how='left', left_index=True, right_index=True)

genders = ['Female', 'Male']
for gender in genders:
    df_this_pop = df_pop[(df_pop['county'] != 'All Combined') &
                          (df_pop['race'] == 'All Combined') &
                          (df_pop['gender'] == gender) &
                          (df_pop['age_group'] == 'All Combined')].drop(['race', 'gender', 'age_group'], axis=1).set_index('county', append=True)
    col_str = 'prop_gender_{}'.format(gender)
    df_this_pop = df_this_pop.rename(columns={'population': col_str})
    df_this_pop[col_str] = df_this_pop[col_str] / df_pop_combo['population']
    
    # join it with the larger df
    df_pop_combo = df_pop_combo.join(df_this_pop, how='left')

races = ['Hispanic', 'Black', 'White', 'Asian/Pacific Islander', 'Native American', 'Other']
for race in races:
    if race == 'Asian/Pacific Islander':
        race_str = 'Asian_PI'
    elif race == 'Native American':
        race_str = 'Native_American'
    else:
        race_str = race
    df_this_pop = df_pop[(df_pop['county'] != 'All Combined') &
                          (df_pop['race'] == race) &
                          (df_pop['gender'] == 'All Combined') &
                          (df_pop['age_group'] == 'All Combined')].drop(['race', 'gender', 'age_group'], axis=1).set_index('county', append=True)
    col_str = 'prop_race_{}'.format(race_str)
    df_this_pop = df_this_pop.rename(columns={'population': col_str})
    df_this_pop[col_str] = df_this_pop[col_str] / df_pop_combo['population']
    
    # join it with the larger df
    df_pop_combo = df_pop_combo.join(df_this_pop, how='left')

df_pop_combo.head()

df_ctx_pop_county = df_ctx_combo.join(df_pop_combo, how='left')

df_ctx_pop_county.head()

# old, not using

# create a state-level view of contextual factors

# # determine the number of people in each county for these columns based on percentage columns and population
# cols = ['less_than_high_school', 'high_school_or_higher', 'bachelors_or_higher',
#         'poverty_rate', 'employment_rate', 'unemployment_rate']

# new_cols = ['population', 'county_income']
# for col in cols:
#     new_col = 'n_{}'.format(col)
#     new_cols.append(new_col)
#     df_ctx_pop_county[new_col] = df_ctx_pop_county[col] * 0.01 * df_ctx_pop_county['population']

# # set up how to aggregate
# agg = {}
# for new_col in new_cols:
#     agg[new_col] = 'sum'
# agg['median_income'] = 'mean'

# # income of the county
# df_ctx_pop_county['county_income'] = df_ctx_pop_county['per_capita_income'] * df_ctx_pop_county['population']

# # get state-level info by grouping all the counties together
# df_ctx_pop_state = df_ctx_pop_county.groupby(level=[0])[new_cols].agg(agg)

# # drop the new columns, except population, because we're going to try joining this to arrests
# df_ctx_pop_county = df_ctx_pop_county.drop([x for x in new_cols if x != 'population'], axis=1)

# # turn state numbers into percentages
# for new_col in [x for x in new_cols if x != 'population']:
#     if new_col != 'county_income':
#         col = new_col.replace('n_', '')
#         df_ctx_pop_state[col] = (df_ctx_pop_state[new_col] / df_ctx_pop_state['population']) * 100
#     else:
#         df_ctx_pop_state['per_capita_income'] = (df_ctx_pop_state[new_col] / df_ctx_pop_state['population'])

# # keep only these columns
# keep_cols = ['population', 'per_capita_income', 'median_income',
#              'less_thahigh_school', 'high_school_or_higher', 'bachelors_or_higher',
#              'poverty_rate', 'employment_rate', 'unemployment_rate']
# df_ctx_pop_state = df_ctx_pop_state[keep_cols]

# df_ctx_pop_state.head()

# Load the fixed offense codes file

df_off = pd.read_csv('offense_codes_v2.csv', quotechar='"')
df_off.head()

# Load the arrests for a given year
dtype={'county': str, 'agency_name': str, 'agency_code': str,
       'arrest_year': int, 'arrest_month': int, 'arrest_day': int,
       'race_or_ethnicity': str, 'gender': str, 'age_group': str,
       'summary_offense_level': str,'offense_level': str,'bcs_offense_code': int,
       'bcs_summary_offence_code': int,'fbi_offense_code': str,'status_type': str,
       'disposition': str}

# for year in pd.period_range(start='2009', end='2014', freq='A'):
year = '2009'

filename = 'ca_doj_arrests_deidentified_2000-2014_05-07-2016/ca_doj_arrests_deidentified_{year}_05-07-2016.csv'.format(year=year)

df_arr = pd.read_csv(filename, dtype=dtype)

df_arr.head()

df_arr['age_group'].value_counts()

df_arr['gender'].value_counts()

df_arr['race_or_ethnicity'].value_counts()

# shorten race strings
df_arr.loc[df_arr['race_or_ethnicity'] == 'suppressed_due_to_privacy_concern', 'race_or_ethnicity'] = 'suppressed'
df_arr.loc[df_arr['race_or_ethnicity'] == 'Asian/Pacific Islander', 'race_or_ethnicity'] = 'Asian_PI'

df_arr[df_arr['age_group'] == 'juvenile']['offense_level'].value_counts()

# overall, juveniles are cited and released a lot more than adults
# (they probably commit different types of crimes)

print('overall')
print(df_arr[(df_arr['age_group'] == 'juvenile')]['status_type'].value_counts().sort_index())
print(df_arr[(df_arr['age_group'] == 'adult')]['status_type'].value_counts().sort_index())

print('\nfelony')
print(df_arr[(df_arr['age_group'] == 'juvenile') & (df_arr['offense_level'] == 'felony')]['status_type'].value_counts().sort_index())
print(df_arr[(df_arr['age_group'] == 'adult') & (df_arr['offense_level'] == 'felony')]['status_type'].value_counts().sort_index())

print('\nmisdemeanor')
print(df_arr[(df_arr['age_group'] == 'juvenile') & (df_arr['offense_level'] == 'misdemeanor')]['status_type'].value_counts().sort_index())
print(df_arr[(df_arr['age_group'] == 'adult') & (df_arr['offense_level'] == 'misdemeanor')]['status_type'].value_counts().sort_index())

# juvenile felonies lead to bookings at a much higher rate than misdemeanors
# (makes sense, they're quite different crimes)

print(df_arr[(df_arr['age_group'] == 'juvenile') & (df_arr['offense_level'] == 'felony')]['status_type'].value_counts().sort_index())

print(df_arr[(df_arr['age_group'] == 'juvenile') & (df_arr['offense_level'] == 'misdemeanor')]['status_type'].value_counts().sort_index())

print(df_arr[(df_arr['age_group'] == 'juvenile') & (df_arr['offense_level'] == 'felony')]['disposition'].value_counts().sort_index())
print(df_arr[(df_arr['age_group'] == 'juvenile') & (df_arr['offense_level'] == 'misdemeanor')]['disposition'].value_counts().sort_index())
print(df_arr[(df_arr['age_group'] == 'juvenile') & (df_arr['offense_level'] == 'status offense')]['disposition'].value_counts().sort_index())

# fix wonky dates

# fix rows with arrest_day == 0
df_arr.loc[(df_arr['arrest_day'] == 0), 'arrest_day'] = 1

# fix months with 30 days that have arrest_day == 31
month_30 = [4, 6, 9, 11]
for month in month_30:
    df_arr.loc[(df_arr['arrest_month'] == month) & (df_arr['arrest_day'] == 31), 'arrest_day'] = 30

# Roll February arrest_day past 28 (i.e., 29, 30, 31) or 29th (i.e., 30, 31) back to max number of days
df_arr.loc[(df_arr['arrest_year'].apply(lambda x: isleap(x))) & (df_arr['arrest_month'] == 2) & (df_arr['arrest_day'] > 29), 'arrest_day'] = 29
df_arr.loc[~(df_arr['arrest_year'].apply(lambda x: isleap(x))) & (df_arr['arrest_month'] == 2) & (df_arr['arrest_day'] > 28), 'arrest_day'] = 28

# # Convert to datetime
# df_arr['arrest_year'] = df_arr['arrest_year'].astype(str)
# df_arr['arrest_month'] = df_arr['arrest_month'].apply(lambda x: '{:02d}'.format(x))
# df_arr['arrest_day'] = df_arr['arrest_day'].apply(lambda x: '{:02d}'.format(x))
# df_arr['arrest_date'] = df_arr[['arrest_year', 'arrest_month', 'arrest_day']].apply(lambda x: pd.Timestamp('-'.join(x)), axis=1)

df_arr['arrest_date'] = pd.to_datetime(df_arr['arrest_year']*10000 + df_arr['arrest_month']*100 + df_arr['arrest_day'], format="%Y%m%d")

# drop the columns for assembling the date
df_arr = df_arr.drop(['arrest_year', 'arrest_month', 'arrest_day'], axis=1)

df_arr = df_arr.set_index('arrest_date')

# limit using the criteria for this question

df_fel = df_arr[(df_arr['age_group'] == 'juvenile') &
                (df_arr['offense_level'] == 'felony')
               ]
print(df_fel.shape)

df_fel['bcs_summary_offence_code'].value_counts().head(10)

df_fel['bcs_offense_code'].value_counts().head(10)

# investigate some of the top offenses
df_off[df_off['bcs_offense_code'] == 400].head()

# investigate some of the top offenses
df_off[df_off['bcs_summary_offence_code'] == 6].head()

# add the county contextual numbers to the felony data frame
df_fel.loc[:, 'year'] = df_fel.index.year

df_fel = df_fel.merge(df_ctx_pop_county, how='left', left_on=['year', 'county'], right_index=True)

df_fel = df_fel.drop(['year'], axis=1)

# code the outcome variable
df_fel.loc[:, 'booked'] = np.nan
df_fel.loc[(df_fel['status_type'] == 'booked'), 'booked'] = 1
df_fel.loc[(df_fel['status_type'] == 'cited'), 'booked'] = 0
# df_fel.loc[(df_fel['status_type'] == 'other'), 'booked'] = np.nan

# 81% of juvenile felony arrests are booked, vs cited and released

df_fel[(df_fel['booked'] == 0) | (df_fel['booked'] == 1)]['booked'].mean()

df_fel.head()

df_fel.columns

def get_sorted_title_offense(df_off, code, n_title_words=4):
    list_of_lists = df_off[df_off['bcs_offense_code'] == code]['description'].apply(lambda x: re.sub('[^A-Za-z]+', ' ', x).split()).values.tolist()
    lst = [val for sublist in list_of_lists for val in sublist]
    lst_sorted=sorted([ss for ss in set(lst) if len(ss)>3], 
                       key=lst.count, 
                       reverse=True)
    lst_sorted = [x for x in lst_sorted if x.upper() != 'ETC']
    title_str = ', '.join(lst_sorted[:n_title_words]).title()
    return title_str

n_codes = 12

n_title_words=4

ncols=3
nrows=int(np.ceil(n_codes / ncols))

field = 'race_or_ethnicity'

fig, (axes) = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 5*nrows))

for ax, code in zip(axes.flat, df_fel['bcs_offense_code'].value_counts().head(n_codes).index.tolist()):
    code_count = df_fel[df_fel['bcs_offense_code'] == code].groupby(by=[field, 'status_type']).size()
    race_count = df_fel[df_fel['bcs_offense_code'] == code].groupby(by=[field]).size()

    prop_race = pd.DataFrame(code_count / race_count).unstack()[0]
    #prop_race = prop_race.sort_values(by='booked', ascending=False)

    title_str = get_sorted_title_offense(df_off, code, n_title_words=n_title_words)

    prop_race.plot(ax=ax, kind='bar');
    
    xlabels = ['{r}\nn={n}'.format(r=x.get_text(), n=race_count[x.get_text()]) for x in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels, rotation=45)
    
    ax.set_ylim((0, 1));
    ax.set_ylabel('Proportion');
    ax.set_title('{c}: {t}'.format(c=code,t=title_str));
    ax.legend(bbox_to_anchor=(1.26, 1.0));
    
fig.tight_layout();

n_codes = 12

n_title_words=4

ncols=3
nrows=int(np.ceil(n_codes / ncols))

field = 'gender'

fig, (axes) = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols, 5*nrows))

for ax, code in zip(axes.flat, df_fel['bcs_offense_code'].value_counts().head(n_codes).index.tolist()):
    code_count = df_fel[df_fel['bcs_offense_code'] == code].groupby(by=[field, 'status_type']).size()
    race_count = df_fel[df_fel['bcs_offense_code'] == code].groupby(by=[field]).size()

    prop_race = pd.DataFrame(code_count / race_count).unstack()[0]
    #prop_race = prop_race.sort_values(by='booked', ascending=False)

    title_str = get_sorted_title_offense(df_off, code, n_title_words=n_title_words)

    prop_race.plot(ax=ax, kind='bar');
    
    xlabels = ['{r}\nn={n}'.format(r=x.get_text(), n=race_count[x.get_text()]) for x in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels, rotation=45)
    
    ax.set_ylim((0, 1));
    ax.set_ylabel('Proportion');
    ax.set_title('{c}: {t}'.format(c=code,t=title_str));
    ax.legend(bbox_to_anchor=(1.24, 1.0));
    
fig.tight_layout();

# cols = ['less_than_high_school',
#         'high_school_or_higher',
#         'bachelors_or_higher',
#         'per_capita_income',
#         'median_income',
#         'poverty_rate',
#         'employment_rate',
#         'unemployment_rate',
#         'population',
#        ]

# formula = 'booked ~ C(race_or_ethnicity) + C(gender) + {}'.format(' + '.join(cols))
# print(formula)
# mod = smf.logit(formula, df_fel)
# results = mod.fit()
# results.summary2()

# cols = ['less_than_high_school',
#         'high_school_or_higher',
#         'bachelors_or_higher',
#         'per_capita_income',
#         'median_income',
#         'poverty_rate',
#         'employment_rate',
#         'unemployment_rate',
#         'population',
#        ]

# formula = 'booked ~ C(race_or_ethnicity) + C(gender) + {}'.format(' + '.join(cols))

# results = cu.sm_logit(df_fel,
#                       f=formula,
#                       add_constant=True,
#                       #features=features,
#                       #categorical=categorical,
#                       #outcome=outcome,
#                      )

# # formula = 'booked ~ C(race_or_ethnicity) + C(gender) + {}'.format(' + '.join(cols))

# outcome = 'booked'
# # outcome = 'offense_level' # for felony vs misdemeanor, not working on this yet

# features = ['race_or_ethnicity',
#             'gender',
#             'bcs_offense_code',
#             #'bcs_summary_offence_code',
#             #'fbi_offense_code',
#             #'status_type',
#             'disposition',
#             'booked',
#             'less_than_high_school',
#             'high_school_or_higher',
#             'bachelors_or_higher',
#             #'per_capita_income', # not inclding because could be confounded with county population
#             'median_income',
#             'poverty_rate',
#             'employment_rate',
#             'unemployment_rate',
#             'population',
#            ]
# categorical = ['race_or_ethnicity',
#                'gender',
#                'bcs_offense_code',
#                #'bcs_summary_offence_code',
#                #'fbi_offense_code',
#                #'status_type',
#                'disposition',
#               ]

# results = cu.sm_logit(df_fel,
#                       #f=formula,
#                       subset=(df_fel['booked'] == 0) | (df_fel['booked'] == 1),
#                       add_constant=True,
#                       features=features,
#                       categorical=categorical,
#                       outcome=outcome,
#                       reg_method='l1',
#                       reg_alpha=1.0,
#                       missing='raise',
#                      )

# set up the formula

f = '''booked ~ 1
+ C(race_or_ethnicity)
+ C(gender)
+ C(race_or_ethnicity):C(gender)
+ less_than_high_school
+ high_school_or_higher
+ bachelors_or_higher
+ poverty_rate
+ employment_rate
+ unemployment_rate
+ I(median_income / 10000)
+ I(population / 10000)
+ I(prop_age_Adult * 100)
+ I(prop_gender_Male * 100)
+ I(prop_race_White * 100)
+ I(prop_race_Black * 100)
+ I(prop_race_Hispanic * 100)
'''

# + C(bcs_offense_code)

# + center(median_income)
# + center(population)

# + I(center(median_income) / 10000)
# + I(center(population) / 10000)

# + I(prop_race_Asian_PI * 100)
# + I(prop_race_Native_American * 100)
# + I(prop_race_Other * 100)

# + I(prop_age_Juvenile * 100)
# + I(prop_gender_Female * 100)

# + C(disposition)

# '+ I(per_capita_income / 10000)' # not inclding because could be confounded with county population

#  & (df_fel['bcs_offense_code'] == 300)

subset = ((df_fel['booked'] == 0) | (df_fel['booked'] == 1))

results = cu.sm_logit(df_fel,
                      f=f,
                      subset=subset,
                      #add_constant=True,
                      #features=features,
                      #categorical=categorical,
                      #outcome=outcome,
                      reg_method='l1',
                      reg_alpha=10.0,
                      missing='raise',
                      maxiter=100,
                     )

cu.print_sm_logit_results(results,
                          print_n=40,
                          print_p_limit=.05,
                          outcome_behavior='to be booked vs. cited and released at the time of arrest');

# found when including bcs_offense_code as categorical feature in logistic regression

# makes sense that murder-related offenses is highly associated with bookings!

# one way to get a summary of this offense code
code=300
n_title_words=6
print(get_sorted_title_offense(df_off, code, n_title_words=n_title_words))

# or view them all
df_off[(df_off['bcs_offense_code'] == code)].head()

# found when including bcs_offense_code as categorical feature in logistic regression

code=410
n_title_words=6
print(get_sorted_title_offense(df_off, code, n_title_words=n_title_words))

df_off[(df_off['bcs_offense_code'] == code)].head()

# found when including bcs_offense_code as categorical feature in logistic regression

code=993
n_title_words=6
print(get_sorted_title_offense(df_off, code, n_title_words=n_title_words))

# or view them all
df_off[(df_off['bcs_offense_code'] == code)].head()

# fig, ax = plt.subplots()

# field = 'high_school_or_higher'

# ax.plot(df_fel.loc[subset, :][field], cu.binary_jitter(df_fel.loc[subset, :]['booked'], .1), '.', alpha = .1)
# ax.plot(np.sort(df_fel.loc[subset, :][field]), results.predict()[np.argsort(df_fel.loc[subset, :][field])], lw = 2)
# ax.set_ylabel('Booked');
# ax.set_xlabel(field);

# field = 'high_school_or_higher'

# field = 'median_income'

# fig, ax = plt.subplots()

# df_fel.loc[subset & (df_fel['booked'] == 1), :][field].plot(ax=ax, kind='kde', label='Booked')
# df_fel.loc[subset & (df_fel['booked'] == 0), :][field].plot(ax=ax, kind='kde', label='Cited')

# ax.legend(loc='best');
# ax.set_xlabel(field);

# # ax.set_xlim(left=60, right=100);

# Run one logistic model for each of the top offense codes

f = '''booked ~ C(race_or_ethnicity) + C(gender)
+ C(race_or_ethnicity):C(gender)
+ less_than_high_school + high_school_or_higher + bachelors_or_higher
+ poverty_rate + employment_rate + unemployment_rate
+ I(median_income / 10000)
+ I(population / 10000)
+ I(prop_age_Adult * 100)
+ I(prop_gender_Male * 100)
+ I(prop_race_White * 100)
'''
# + C(bcs_offense_code)

# + I(prop_age_Juvenile * 100)
# + I(prop_gender_Female * 100)
# + I(prop_race_Hispanic * 100)
# + I(prop_race_Black * 100)
# + I(prop_race_Asian_PI * 100)
# + I(prop_race_Native_American * 100)
# + I(prop_race_Other * 100)

# + C(disposition)

# '+ I(per_capita_income/10000)' # not inclding because could be confounded with county population

n_codes=12
n_title_words=5

for code in df_fel['bcs_offense_code'].value_counts().index[:n_codes].tolist():
    title_str = get_sorted_title_offense(df_off, code, n_title_words=n_title_words)
    print('========================================')
    print('{c}: {t}'.format(c=code, t=title_str))
    print('========================================')
    
    subset = ((df_fel['booked'] == 0) | (df_fel['booked'] == 1)) & (df_fel['bcs_offense_code'] == code)
    
    results = cu.sm_logit(df_fel,
                          f=f,
                          subset=subset,
                          #add_constant=True,
                          #features=features,
                          #categorical=categorical,
                          #outcome=outcome,
                          reg_method='l1',
                          reg_alpha=0.1,
                          missing='raise',
                         )

    cu.print_sm_logit_results(results,
                              print_n=40,
                              print_p_limit=.05,
                              outcome_behavior='to be booked for {t} vs. cited and released at the time of arrest'.format(t=title_str));

# field = 'median_income'
# # field='prop_gender_Male'
# field='poverty_rate'
# bins=6

# resids = cu.bin_residuals(results.resid_response,
#                                df_fel.loc[subset, :][field],
#                                bins)
# plt.figure(figsize = (6, 5))
# plt.ylabel('Residual (bin avg.)')
# plt.xlabel('{} (bin avg.)'.format(field))
# cu.plot_binned_residuals(resids)

fields = ['less_than_high_school',
        'high_school_or_higher',
        'bachelors_or_higher',
        'per_capita_income',
        'median_income',
        'poverty_rate',
        'employment_rate',
        'unemployment_rate',
        'population',
         ]
cu.plot_hist(df_fel[(df_fel['booked'] == 1)], df_fel[(df_fel['booked'] == 0)],
             results=fields,
             normed=True,
             pos_str='Booked',
             neg_str='Cited',
             log_trans=['population']
            );

# #'summary_offense_level', 'offense_level',

# features = ['race_or_ethnicity', 'gender',
#             #'bcs_offense_code',
#             'bcs_summary_offence_code',
#             #'fbi_offense_code',
#             #'status_type',
#             #'disposition',
#             'booked',
#            ]
# categorical = ['race_or_ethnicity', 'gender',
#                #'bcs_offense_code',
#                'bcs_summary_offence_code',
#                #'fbi_offense_code',
#                #'status_type',
#                #'disposition',
#               ]
# # outcome = 'offense_level'
# outcome = 'booked'

# results = cu.sm_logit(df_fel,
#                       add_constant=True,
#                       features=features,
#                       outcome=outcome,
#                       categorical=categorical)

# cu.print_sm_logit_results(results, print_p_limit=.05);

# cu.classify(X_train, X_test, y_train, y_test, classifier='lr')

# for col in df_fel.columns:
#     print('\n',df_fel[col].value_counts().head(10))

# from sklearn.ensemble import RandomForestClassifier

