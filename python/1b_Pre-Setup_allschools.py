import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import presetup as p
import seaborn as sns
import re
from collections import Counter
from sklearn.cross_validation import train_test_split
get_ipython().magic('matplotlib inline')

df = pd.read_csv('../data/raw/raw_data.csv', low_memory=False)

ps = p.PreSetup()
col_dict = ps.parseCols('../data/reference/column_names.txt')
df.columns = ps.updateCols(df.columns.values)

df['Who are you?'].value_counts()

df = df[df['Who are you?']=='Admit Creating College / Grad School Profile'].copy()
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)

vals = ['YTowOnt9', 'ytowont9', 'czowOiIiOw==']
for v in vals:
    df.replace(to_replace=v, value=np.nan, inplace=True)

sc = p.Schools()

all_schools = sc.getSchools('../data/reference/table_references.csv')

print len(all_schools), len(set(all_schools))

all_schools = list(set(all_schools))

df_schools = pd.DataFrame(index=xrange(len(df)), columns=all_schools)

sc.extractFromApplied(df['Undergraduate Schools Applied'], df_schools)

df_schools = df_schools[all_schools]

# top_schools = ['Harvard University (Cambridge, MA)', 'Yale University (New Haven, CT)', 
#                'Cornell University (Ithaca, NY)', 'Columbia University (New York, NY)',
#                'University of Pennsylvania (Philadelphia, PA)', 'Princeton University (Princeton, NJ)',
#                'Brown University (Providence, RI)', 'Dartmouth College (Hanover, NH)',
#                'Massachusetts Institute of Technology (Cambridge, MA)','Stanford University (Stanford, CA)']
# df_topschools = df_schools[top_schools].copy()

for school in all_schools:
    df_schools[school] = df_schools[school].apply(lambda x: sc.cleanFromApplied(x) if type(x) == str else x)

# df_topschools['any_top_school'] = (df_topschools.sum(axis=1)).apply(lambda x: 1 if x>0 else np.nan)

mask_school = df_schools.notnull().sum(axis=0)
df_schools2 = df_schools[mask_school[mask_school>60].index].copy()
df_schools2 = df_schools2.fillna(value=0)

subset_schools = df_schools2.columns

# df_schools2.to_csv('../data/all_schools.csv')

# df = df[df.columns[:-232]].copy()

# Join back with main df
df = df.join(df_schools2)

print df['Undergraduate Schools Applied'].notnull().sum()
print df['Undergraduate Schools Attended'].notnull().sum()

for s in subset_schools:
    df[s+'_v2'] = df['Undergraduate Schools Attended'].apply(lambda x: sc.extractAllFromAttended(x, s))

for s in subset_schools:
    df[s+'_final'] = ((df[s] + df[s+'_v2'])>0).astype(int)

output_cols = [s+'_final' for s in subset_schools]
output_cols.append('id')
df_output = df[output_cols].copy()

df_output.to_csv('../data/all_schools.csv')

