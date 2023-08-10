get_ipython().magic('matplotlib inline')
from __future__ import print_function, division
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from os.path import join as pjoin
sns.set_style('darkgrid')
sns.set_context('poster')

basepath = "C:/WorkSpace/data/TC/pia"
filename = "cairns_impact.csv"

df01 = pd.read_csv(pjoin(basepath, "01", filename), header=0)
df02 = pd.read_csv(pjoin(basepath, "02", filename), header=0)
df03 = pd.read_csv(pjoin(basepath, "03", filename), header=0)

print("Mean losses for three scenarios:")
print("Scenario 1: ${0:,.2f}".format(np.sum(df01['loss'])))
print("Scenario 2: ${0:,.2f}".format(np.sum(df02['loss'])))
print("Scenario 3: ${0:,.2f}".format(np.sum(df03['loss'])))

filename = "cairns_impact_classified_local.csv"

data01 = pd.read_csv(pjoin(basepath, "01", filename), header=0)
slr_cols = [col for col in data01.keys() if col.startswith('structural_loss_ratio') and col[-1].isdigit()]
slrdf01 = data01[slr_cols]
slrdf01.loc[:, '0.2s gust at 10m height m/s'] = data01['0.2s gust at 10m height m/s']
slrdf01.loc[:, 'SUBURB_2015'] = data01['SUBURB_2015']
slrdf01.loc[:, 'LID'] = data01['LID']
slrdf01.loc[:, 'REPLACEMENT_VALUE'] = data01['REPLACEMENT_VALUE']
slrdf01.loc[:, 'M4'] = data01['M4']
slrdf01.loc[:, 'YEAR_BUILT'] = data01['YEAR_BUILT']
sldf01 = slrdf01[['0.2s gust at 10m height m/s', 'SUBURB_2015', 'LID', 'REPLACEMENT_VALUE', 'M4', 'YEAR_BUILT']]
sl_cols = []
for col in slr_cols:
    slcolname = col.replace("_ratio", "")
    sl_cols.append(slcolname)
    sldf01.loc[:, slcolname] = slrdf01[col] * sldf01['REPLACEMENT_VALUE']
    
sumsl01 = np.sum(sldf01.loc[:, sl_cols]).values/10**6
sumslmean01 = np.mean(sumsl01)

data02 = pd.read_csv(pjoin(basepath, "02", filename), header=0)
slr_cols = [col for col in data02.keys() if col.startswith('structural_loss_ratio') and col[-1].isdigit()]
slrdf02 = data02[slr_cols]
slrdf02.loc[:, '0.2s gust at 10m height m/s'] = data02['0.2s gust at 10m height m/s']
slrdf02.loc[:, 'SUBURB_2015'] = data02['SUBURB_2015']
slrdf02.loc[:, 'LID'] = data02['LID']
slrdf02.loc[:, 'REPLACEMENT_VALUE'] = data02['REPLACEMENT_VALUE']
slrdf02.loc[:, 'M4'] = data02['M4']
slrdf02.loc[:, 'YEAR_BUILT'] = data02['YEAR_BUILT']
sldf02 = slrdf02[['0.2s gust at 10m height m/s', 'SUBURB_2015', 'LID', 'REPLACEMENT_VALUE', 'M4', 'YEAR_BUILT']]
sl_cols = []
for col in slr_cols:
    slcolname = col.replace("_ratio", "")
    sl_cols.append(slcolname)
    sldf02.loc[:, slcolname] = slrdf02[col] * sldf02['REPLACEMENT_VALUE']
    
sumsl02 = np.sum(sldf02.loc[:, sl_cols]).values/10**6
sumslmean02 = np.mean(sumsl02)

data03 = pd.read_csv(pjoin(basepath, "03", filename), header=0)
slr_cols = [col for col in data03.keys() if col.startswith('structural_loss_ratio') and col[-1].isdigit()]
slrdf03 = data03[slr_cols]
slrdf03.loc[:, '0.2s gust at 10m height m/s'] = data03['0.2s gust at 10m height m/s']
slrdf03.loc[:, 'SUBURB_2015'] = data03['SUBURB_2015']
slrdf03.loc[:, 'LID'] = data03['LID']
slrdf03.loc[:, 'REPLACEMENT_VALUE'] = data03['REPLACEMENT_VALUE']
slrdf03.loc[:, 'M4'] = data03['M4']
slrdf03.loc[:, 'YEAR_BUILT'] = data03['YEAR_BUILT']
sldf03 = slrdf03[['0.2s gust at 10m height m/s', 'SUBURB_2015', 'LID', 'REPLACEMENT_VALUE', 'M4', 'YEAR_BUILT']]
sl_cols = []
for col in slr_cols:
    slcolname = col.replace("_ratio", "")
    sl_cols.append(slcolname)
    sldf03.loc[:, slcolname] = slrdf03[col] * sldf03['REPLACEMENT_VALUE']
    
sumsl03 = np.sum(sldf03.loc[:, sl_cols]).values/10**6
sumslmean03 = np.mean(sumsl03)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
bins = np.arange(30, 90, 1)
sns.distplot(slrdf01['0.2s gust at 10m height m/s'],
             axlabel="0.2s gust at 10m height m/s", 
             ax=ax, color='Blue', bins=bins,
             hist_kws={'label':'Scenario 1', 'alpha':0.25})

sns.distplot(slrdf02['0.2s gust at 10m height m/s'],
             axlabel="0.2s gust at 10m height m/s", 
             ax=ax, color='Red', bins=bins,
             hist_kws={'label':'Scenario 2', 'alpha':0.25})

sns.distplot(slrdf03['0.2s gust at 10m height m/s'],
             axlabel="0.2s gust at 10m height m/s", 
             ax=ax, color='Green', bins=bins,
             hist_kws={'label':'Scenario 3', 'alpha':0.25})

ax.legend(loc=9, ncol=3)

print(np.median(slrdf01['0.2s gust at 10m height m/s']))
print(np.median(slrdf02['0.2s gust at 10m height m/s']))
print(np.median(slrdf03['0.2s gust at 10m height m/s']))

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
sns.distplot(sumsl01, axlabel="Total structural loss ($ million)", 
             ax=ax, color='Blue',
             hist_kws={'label':'Scenario 1', 'alpha':0.25})
ax.axvline(sumslmean01, label="Mean: ${0:0.1f} million".format(sumslmean01), color='b')

sns.distplot(sumsl02, axlabel="Total structural loss ($ million)", 
             ax=ax, color='Red',
             hist_kws={'label':'Scenario 2', 'alpha':0.25})
ax.axvline(sumslmean02, label="Mean: ${0:0.1f} million".format(sumslmean02), color='r')

sns.distplot(sumsl03, axlabel="Total structural loss ($ million)", 
             ax=ax, color='Green',
             hist_kws={'label':'Scenario 3', 'alpha':0.25})
ax.axvline(sumslmean03, label="Mean: ${0:0.1f} million".format(sumslmean03), color='g')

ax.set_xlim((0, 1500))
ax.legend(loc=2, ncol=2)

def classifyDamageState(df, states, thresholds):

    df['Damage state'] = "Negligible"
    for thres, state in zip(thresholds, states):
        idx = np.where(df['loss_ratio'] >= thres)[0]
        df['Damage state'][idx] = state
        
    return df

thresholds = [0.01, 0.1, 0.2, 0.5]
states = ['Slight', 'Moderate', 'Extensive', 'Complete']
df01 = classifyDamageState(df01, states, thresholds)
df02 = classifyDamageState(df02, states, thresholds)
df03 = classifyDamageState(df03, states, thresholds)

ax = sns.countplot(x='Damage state', data=df01, palette='RdBu', hue='YEAR_BUILT',
                   order=['Negligible', 'Slight', 'Moderate', 'Extensive', 'Complete'])
ax.legend(loc=1)
ax.set_xlabel("Expected damage state")

ax = sns.countplot(x='Damage state', data=df02, palette='RdBu', hue='YEAR_BUILT',
                   order=['Negligible', 'Slight', 'Moderate', 'Extensive', 'Complete'])
ax.legend(loc=1)
ax.set_xlabel("Expected damage state")

ax = sns.countplot(x='Damage state', data=df03, palette='RdBu', hue='YEAR_BUILT',
                   order=['Negligible', 'Slight', 'Moderate', 'Extensive', 'Complete'])
ax.legend(loc=1)
ax.set_xlabel("Expected damage state")

agedmg01 = df01.groupby(['Damage state', 'YEAR_BUILT'])
100 * agedmg01.count()['latitude']/len(df01)

agedmg02 = df02.groupby(['Damage state', 'YEAR_BUILT'])
100 * agedmg02.count()['latitude']/len(df02)

agedmg03 = df03.groupby(['Damage state', 'YEAR_BUILT'])
100 * agedmg03.count()['latitude']/len(df03)

print(agedmg01.count()['latitude'])

print(agedmg02.count()['latitude'])

print(agedmg03.count()['latitude'])

subdmg01 = df01.groupby(['SUBURB_2015', 'Damage state'])
subdmg02 = df02.groupby(['SUBURB_2015', 'Damage state'])
subdmg03 = df03.groupby(['SUBURB_2015', 'Damage state'])
print(subdmg01.count()['latitude'])
print(subdmg02.count()['latitude'])
print(subdmg03.count()['latitude'])

dmg01 = df01.groupby(['Damage state'])
dmg02 = df02.groupby(['Damage state'])
dmg03 = df03.groupby(['Damage state'])

print(dmg01.count()['latitude'])
print(dmg02.count()['latitude'])
print(dmg03.count()['latitude'])


