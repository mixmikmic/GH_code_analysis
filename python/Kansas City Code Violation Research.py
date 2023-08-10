# Code Preparation
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from IPython import display
get_ipython().magic('matplotlib inline')

# Data Preparation
# There's a warning message for 'Inspection Area', as it's not all in integer
# We parse ['Case Opened Date', 'Case Closed Date', 'Violation Entry Date'] to be datetime object
def parse_date(x):
    if x in ['nan', '']:
        return None
    return datetime.strptime(str(x), '%m/%d/%Y')
df = pd.read_csv('Property_Violations.csv',
                    parse_dates=[3,4,10], date_parser=parse_date)
df.drop_duplicates(subset=['Case ID'])
print 'data size', df.shape

# Use only the data from 2009 to 2015
df = df[df['Case Opened Date'].dt.year.isin(range(2009,2016))]

# Let's look at the data size and columns
print df.shape
print df.dtypes

weekday_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
def plot_by_weekday(col):
    g = df[col].groupby(df[col].dt.weekday).count().plot(kind="bar", alpha=0.5, ylim=(0,100000))
    g.set_xticklabels(weekday_labels, rotation=0)
    
plot_by_weekday('Case Opened Date')

plot_by_weekday('Case Closed Date')

def plot_by_month(col):
    g = df[col].groupby(df[col].dt.month).count().plot(kind="bar", alpha=0.5, stacked=True, ylim=(0,60000))
    g.set_xticklabels(range(1,13), rotation=0)
plot_by_month('Case Opened Date')

plot_by_month('Case Closed Date')

top_violations = df.groupby('Violation Code').size().nlargest(10)
print top_violations

dfdesc = df[['Violation Code', 'Violation Description']]
dfdesc = dfdesc[dfdesc['Violation Code'].isin(top_violations.index)]
vdesc = dict(zip(dfdesc['Violation Code'].unique(), dfdesc['Violation Description'].unique()))
for i in top_violations.index:
    print "%12s: %s" % (i, vdesc[i])

df2 = df[['Violation Code', 'Case Opened Date']]
df2 = df2[df2['Violation Code'].isin(top_violations.index)]
df2 = df2.groupby([df['Case Opened Date'].dt.month, 'Violation Code'])['Violation Code'].count().unstack().fillna(0)
df2 = df2[top_violations.index] #Use top violation index for Violation Code order
g = df2.plot(kind='bar',sort_columns=True, stacked=True, figsize=(20,10), fontsize=15, colormap='Set1', ylim=(0,50000))
g.set_xticklabels(range(1,13), rotation=0)
g.plot()

df['Violation Code Purged'] = df['Violation Code'].apply(lambda x: x.rstrip('1234567890.'))
top_violations_purged = df.groupby('Violation Code Purged').size().nlargest(10)
print top_violations_purged

df2 = df[['Violation Code Purged', 'Case Opened Date']]
df2 = df2[df2['Violation Code Purged'].isin(top_violations_purged.index)]
df2 = df2.groupby([df['Case Opened Date'].dt.month, 'Violation Code Purged'])['Violation Code Purged'].count().unstack().fillna(0)
df2 = df2[top_violations_purged.index] #Use top violation index for Violation Code order
g = df2.plot(kind='bar',sort_columns=True, stacked=True, figsize=(20,10), fontsize=15, colormap='Set1', ylim=(0,50000))
g.set_xticklabels(range(1,13), rotation=0)
g.plot()

df5 = df[['Violation Code Purged', 'Days Open']]
df5.groupby('Violation Code Purged').mean().sort_values(by='Days Open', ascending=False)[:20]

#How do the different types of violations among most reported violation types affect how long a case takes to close?
df5 = df[['Violation Code Purged', 'Days Open']]
df6 = df5[df5['Violation Code Purged'].isin(top_violations_purged.index)]
df6.groupby('Violation Code Purged').boxplot(figsize=(20,20))
plt.show()
df6.groupby('Violation Code Purged').mean().sort_values(by='Days Open', ascending=False)

#Violation happens at the same location
from itertools import combinations
dfto = df[['Violation Code', 'Code Violation Location']]
dftolsize = dfto.groupby('Code Violation Location').size()
mulcv_locations = dftolsize[dftolsize > 1].index
mdfto = dfto[dfto['Code Violation Location'].isin(mulcv_locations)] #contains locations with multiple cv reported
grouped = mdfto.groupby('Code Violation Location')
cooccur_dict = {}
for name, group in grouped:
    codes = group['Violation Code'].copy()
    for a, b in combinations(codes, 2):
        sorted_comb = sorted((a,b))
        k = " - ".join(sorted_comb)
        if cooccur_dict.has_key(k):
            cooccur_dict[k] += 1
        else:
            cooccur_dict[k] = 1
cooc = pd.DataFrame(cooccur_dict.items(), columns=["Violation Code Combos", "Co-occurance"])
cooc.nlargest(20, "Co-occurance").reset_index(drop=True)

# 3 Combos
grouped = mdfto.groupby('Code Violation Location')
cooccur_dict = {}
for name, group in grouped:
    codes = group['Violation Code'].copy()
    for a, b, c in combinations(codes, 3):
        sorted_comb = sorted((a,b,c))
        k = " - ".join(sorted_comb)
        if cooccur_dict.has_key(k):
            cooccur_dict[k] += 1
        else:
            cooccur_dict[k] = 1
cooc = pd.DataFrame(cooccur_dict.items(), columns=["Violation Code Combos", "Co-occurance"])
cooc.nlargest(20, "Co-occurance").reset_index(drop=True)

# Violation presence and consequences
from itertools import combinations, product
dftod = df[['Violation Code', 'Code Violation Location', 'Case Opened Date']]
dftodlsize = dftod.groupby('Code Violation Location').size()
dftodlsize
mulcv_locations = dftodlsize[dftodlsize > 1].index
mdfto = dftod[dftod['Code Violation Location'].isin(mulcv_locations)] #contains locations with multiple cv reported
grouped = mdfto.groupby(['Code Violation Location'])
progress = 0 
cooccur_dict = {}
for name, group in grouped:
    if len(group['Case Opened Date'].unique()) > 1:
        #reported in multiple dates
        codes_sequences = [g['Violation Code'] for n, g in group.groupby('Case Opened Date')][:6]
        #produce all combination
        for code_sequence in list(product(*codes_sequences)):
            k = " -> ".join(code_sequence)
            if cooccur_dict.has_key(k):
                cooccur_dict[k] += 1
            else:
                cooccur_dict[k] = 1
    progress += 1
    if progress % 1000 == 0: 
        print '=',
#     if dead > 50000: break
cooc = pd.DataFrame(cooccur_dict.items(), columns=["Violation Effects", "Occurance"])
cooc.nlargest(30, "Occurance").reset_index(drop=True)



