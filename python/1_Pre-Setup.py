import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import presetup
import seaborn as sns
import re
from collections import Counter
from sklearn.cross_validation import train_test_split
get_ipython().magic('matplotlib inline')

df = pd.read_csv('../data/raw/raw_data.csv', low_memory=False)

print df.columns.values

ps = presetup.PreSetup()
col_dict = ps.parseCols('../data/reference/column_names.txt')
df.columns = ps.updateCols(df.columns.values)

print df.columns.values

df['Who are you?'].value_counts()

df = df[df['Who are you?']=='Admit Creating College / Grad School Profile'].copy()
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)

vals = ['YTowOnt9', 'ytowont9', 'czowOiIiOw==']
for v in vals:
    df.replace(to_replace=v, value=np.nan, inplace=True)

sc = presetup.Schools()

all_schools = sc.getSchools('../data/reference/table_references.csv')

print len(all_schools), len(set(all_schools))

all_schools = list(set(all_schools))

df_schools = pd.DataFrame(index=xrange(len(df)), columns=all_schools)

sc.extractFromApplied(df['Undergraduate Schools Applied'], df_schools)

df_schools = df_schools[all_schools]

top_schools = ['Harvard University (Cambridge, MA)', 'Yale University (New Haven, CT)', 
               'Cornell University (Ithaca, NY)', 'Columbia University (New York, NY)',
               'University of Pennsylvania (Philadelphia, PA)', 'Princeton University (Princeton, NJ)',
               'Brown University (Providence, RI)', 'Dartmouth College (Hanover, NH)',
               'Massachusetts Institute of Technology (Cambridge, MA)','Stanford University (Stanford, CA)']
df_topschools = df_schools[top_schools].copy()

for school in top_schools:
    df_topschools[school] = df_topschools[school].apply(lambda x: sc.cleanFromApplied(x) if type(x) == str else x)

df_topschools['any_top_school'] = (df_topschools.sum(axis=1)).apply(lambda x: 1 if x>0 else np.nan)

# Join df_topschools back with main df
df = df.join(df_topschools)

print df['Undergraduate Schools Applied'].notnull().sum()
print df['Undergraduate Schools Attended'].notnull().sum()

df['any_top_school_v2'] =  df['Undergraduate Schools Attended'].apply(sc.extractFromAttended)

df['top_school_final'] = df.apply(sc.finalTopSchool, axis=1)

df2 = df[df['Internal Use - Calculated Undergrad Price']>4].copy()

target_distr = df2['top_school_final'].value_counts()
print target_distr

print target_distr/float(sum(target_distr))

df_train, df_test = train_test_split(df, train_size=0.7, random_state=123)

df_train.to_csv('../data/train.csv')
df_test.to_csv('../data/test.csv')
df.to_csv('../data/master.csv')

