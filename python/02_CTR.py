from itertools import combinations
from __future__ import division

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scis

get_ipython().magic('matplotlib inline')

alpha = 0.05/23

alpha

df = pd.read_csv('nyt1.csv')

df.describe().T

df.info()

df.Impressions.value_counts() # An impression (in the context of online advertising) is when an ad is fetched from its source, and is countable. Whether or not the ad is clicked is not taken into account.[1] Each time an ad is fetched it is counted as one impression.[2]

df = df[df.Impressions != 0]

df.Impressions.value_counts()

df['CTR'] = df.Clicks / df.Impressions.astype(float)

df[:3]

df.hist(figsize=(12,8));

hecf = df.hist(figsize=(12,8), grid=False, normed=True, color='b', alpha=.2);
plt.suptitle('Overall');

def plot_hist(df, title, color):
    df.hist(figsize=(12, 8), grid=False, normed=True, color=color, alpha=.2)
    plt.suptitle(title, size=18, weight='bold', y=1.05) #to place a title in the center

plot_hist(df, 'Overall', 'b')

df_signedin = df[df.Signed_In == 1]

df_signedin[:2]

df_signedin.hist(figsize=(12, 8));

plot_hist(df_signedin, 'Signed In', 'g')

df_notsignedin = df[df.Signed_In != 1]

df_notsignedin[:2]

df_notsignedin.hist(figsize=(12, 8));

plot_hist(df_notsignedin, 'NOT Signed In', 'r')

scis.ttest_ind(df_signedin.CTR,df_notsignedin.CTR, equal_var=False)

df_signedin.CTR.mean()

df_notsignedin.CTR.mean()

def welch_ttest(df1, df2, name_1, name_2):
    mean_signedin = df1.CTR.mean()
    mean_notsignedin = df2.CTR.mean()
    
    
    print '{} Mean CTR {}'.format(name_1, mean_signedin)
    print '{} Mean CTR {}'.format(name_2, mean_notsignedin)
    print 'Difference in means:', abs(mean_signedin - mean_notsignedin)
    
    p_val = scis.ttest_ind(df1['CTR'], df2['CTR'], equal_var = False)[1]
    print 'P-value:', p_val
    
    df1['CTR'].hist(normed = True, label=name_1, color='g', alpha=0.3)
    plt.axvline(mean_signedin, color='r', lw=1)
    df2['CTR'].hist(normed = True, label=name_2, color='r', alpha=0.3)
    plt.axvline(mean_notsignedin, color='b', lw=1)
    
    plt.ylabel('Probability Density')
    plt.xlabel('CTR')
    plt.grid('off')
    plt.legend()
    
    
welch_ttest(df_signedin, df_notsignedin, 'Signed in', 'Not Signed in')

df_signedin.Gender.value_counts()

male = df_signedin[df_signedin.Gender == 1]
female = df_signedin[df_signedin.Gender != 1]

welch_ttest(male, female, 'male', 'female');

df_signedin.Age.value_counts(sort=True, ascending=False)

df_signedin['AgeGroup'] = pd.cut(df_signedin.Age, [7, 18, 24, 34, 44, 54, 64, 1000])

df_signedin.AgeGroup

df_signedin.AgeGroup.value_counts().sort_index().plot(kind='bar', grid=False);
plt.xlabel('Age group');
plt.ylabel('Number of users');

#generate the combinations of the pair age group
pairs = combinations(pd.unique(df_signedin.AgeGroup), 2)
print list(pairs)

results = pd.DataFrame()
pairs = combinations(pd.unique(df_signedin.AgeGroup), 2)

for age_1, age_2 in pairs:
    CTR_1 = df_signedin[df_signedin.AgeGroup == age_1]['CTR']
    CTR_2 = df_signedin[df_signedin.AgeGroup == age_2]['CTR']
    p_val = scis.ttest_ind(CTR_1, CTR_2, equal_var=False)[1]
    CTR_m1 = CTR_1.mean()
    CTR_m2 = CTR_2.mean()
    difference = abs(CTR_m1 - CTR_m2)
    results = results.append(dict(one=age_1, two=age_2, mean1=CTR_m1, mean2=CTR_m2, diff=difference, p=p_val), ignore_index=True)

    
results[['one', 'two', 'mean1', 'mean2', 'diff', 'p']]

results[results['p'] < alpha].sort('diff', ascending=False)

results = pd.DataFrame()
pairs = combinations(pd.unique(df_signedin.AgeGroup), 2)

for age_1, age_2 in pairs:
    CTR_1 = df_signedin[df_signedin.AgeGroup == age_1].CTR
    CTR_2 = df_signedin[df_signedin.AgeGroup == age_2].CTR
    p_val = scis.ttest_ind(CTR_1, CTR_2, equal_var=False)[1]
    CTR_m1 = CTR_1.mean()
    CTR_m2 = CTR_2.mean()
    difference = abs(CTR_m1 - CTR_m2)
    results = results.append(dict(one=age_1, two=age_2, mean1=CTR_m1, mean2=CTR_m2, diff=difference, p=p_val), ignore_index=True)

    
results = results[['one', 'two', 'mean1', 'mean2', 'diff', 'p']] # Order of the columns

results[results['p'] < alpha].sort('diff', ascending=False)



























