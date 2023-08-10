import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pylab import *

import igraph as ig

import sys
sys.path.append('../../src/')
from utils.database import dbutils

conn = dbutils.connect()
cursor = conn.cursor()

df = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
df.head()

def frequency(dataframe,columnname):
    out = dataframe[columnname].value_counts().to_frame()
    out.columns = ['frequency']
    out.index.name = columnname
    out.reset_index(inplace=True)
    out = out.sort_values(columnname)
    out['cumulative'] = out['frequency'].cumsum()/out['frequency'].sum()
    out['ccdf'] = 1 - out['cumulative']
    return out

(df['adults_first_use'] + df['adults_reuse'] != df['total_adults']).sum() # Check to make sure the columns add up

(df['total_adults'] > 1).sum() # Check to make sure there is never more than 1 adult per card, acc

fr1 = frequency(df,'minors')
fr1.head() # Only 1 child max per card, which is about 10% of the cases

# Now, do some people visit the same museum more than once?
fr2 = frequency(df.groupby(['museum_name','user_id'])['total_adults'].count().to_frame(),'total_adults')
fr2.head(20) # Yes! Some up to 20?!? Let's investigate more.

df1 = df.groupby(['museum_name','user_id'])['total_adults'].count().to_frame()
df1[df1['total_adults']>5] 

# For Battistero di San Giovanni, it is expected, since that includes all Duomo locations. 
# But for others it's unexpected. Let's take a look at the most egregious case.
df[(df['user_id']==2068648) & (df['museum_name']=='Galleria degli Uffizi')]

# Suggests we need deduplication. But let's make sure.
df2 = df.groupby(['user_id','museum_name','entry_time']).count()
df2[(df2['total_adults']>1)|(df2['minors']>1)].head(50)

df3 = df2[(df2['total_adults']>1)|(df2['minors']>1)]
df3.head()

df3.shape # How many unique user_id / museum_name / entry_time rows

df3['adults_first_use'].sum() # How many entries there are

df3['adults_first_use'].sum() - df3.shape[0] # How many, then, are greater than 1: number of duplicate records

df.shape # How many records there were initially

df.shape[0] - (df3['adults_first_use'].sum() - df3.shape[0]) # Should be the result when deduplicated

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(fr1['minors']+1,fr1['frequency'], linestyle='steps--')
yscale('log')
xscale('log')
ax.set_title('Distribution of number of minors')
ax.set_ylabel('Entrances')
ax.set_xlabel('Number of minors')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.plot(frc2['calls'],frc2['cumulative'])
# yscale('log')
xscale('log')
# ylim([.7,1.01])
ax.set_title('CDF of calls per person among Italians who have made calls in Florence city')
ax.set_ylabel('Proportion of users making x or fewer calls')
ax.set_xlabel('Number of calls')
# axvline(4.5) # Our cutoff
# axhline(.1)
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.plot(frc2['calls'],frc2['ccdf'])
yscale('log')
xscale('log')
ax.set_title('CCDF of calls per person among Italians who have made calls in Florence city')
ax.set_ylabel('Proportion of users making at least x calls')
ax.set_xlabel('Number of calls')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(frc_uc['calls_in_florence_comune'],frc_uc['frequency'], linestyle='steps--')
yscale('log')
xscale('log')
ax.set_title('Calls per person among Italians in Florence city')
ax.set_ylabel('Number of people making x calls')
ax.set_xlabel('Number of calls')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.plot(frc_uc['calls_in_florence_comune'],frc_uc['cumulative'])
# yscale('log')
xscale('log')
ylim([.75,1.01])
ax.set_title('CDF of calls per person among Italians in Florence city')
ax.set_ylabel('Proportion of users making x or fewer calls')
ax.set_xlabel('Number of calls')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.plot(frc_uc['calls_in_florence_comune'],frc_uc['ccdf'])
yscale('log')
xscale('log')
ax.set_title('CCDF of calls per person among Italians in Florence city')
ax.set_ylabel('Proportion of users making at least x calls')
ax.set_xlabel('Number of calls')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(frc_dc['days_active_in_florence_comune'],frc_dc['frequency'], linestyle='steps--')
yscale('log')
xscale('log')
ax.set_title('Days active per person among Italians in Florence city')
ax.set_ylabel('Number of people with x days active')
ax.set_xlabel('Days active')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.plot(frc_dc['days_active_in_florence_comune'],frc_dc['cumulative'])
# yscale('log')
xscale('log')
ylim([.86,1.01])
ax.set_title('CDF of days active per person among Italians in Florence city')
ax.set_ylabel('Proportion of users active on x or fewer days')
ax.set_xlabel('Number of days active')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.plot(frc_dc['days_active_in_florence_comune'],frc_dc['ccdf'])
yscale('log')
xscale('log')
ax.set_title('CCDF of days active per person among Italians in Florence city')
ax.set_ylabel('Proportion of users active on at least x days')
ax.set_xlabel('Number of days a')
plt.show()

dfc['mean_calls_per_day'] = dfc['calls']/dfc['days_active']
dfc[dfc['calls_in_florence_comune']>0].head()

print dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].max()
print dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].min()
print dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].mean()
print dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].median()
print dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].std()
print dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].std()*2+dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].mean()
print dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].std()*3+dfc[dfc['calls_in_florence_comune']>0]['mean_calls_per_day'].mean()

dfc[(dfc['calls_in_florence_comune']>0)&(dfc['mean_calls_per_day']<1000)].plot.hist(y='mean_calls_per_day', logy=True, figsize=(15,10), bins=200)
plt.ylabel('Frequency')
plt.xlabel('Average calls per active day')
plt.axvline(150,color="black")
# plt.xlim([0,1000])
plt.title('Average calls per active day by foreign SIM cards who were in Florence city')

# dfc.plot.scatter(x='calls',y='days_active',figsize=(15,10),logy=True,logx=True)

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=dfc['calls'],y=dfc['days_active'],s=.1)
yscale('log')
xscale('log')
ax.set_title('Calls by days active among Italians')
ax.set_xlabel('Calls')
ax.set_ylabel('Days active')
y=[1/10000, 1*200]
x2=[150/10000, 150*200]
plt.plot(x2,y,color='black',linewidth=.25,ls='dashed')
ax.axvline(4.5,color='black',linewidth=.25,ls='dashed')
ylim([.9,90])
xlim([.9,10*10000])
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=dfc['calls'],y=dfc['calls_in_florence_comune'],s=.1)
yscale('log')
xscale('log')
ax.set_title('Total calls vs calls in Florence city, for Italians with calls in Florence')
ax.set_xlabel('Total calls')
ax.set_ylabel('Calls in Florence city')
xlim([.9,10*10000])
ylim([.9,10*10000])
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=dfc['days_active'],y=dfc['days_active_in_florence_comune'],s=.1)
yscale('log')
xscale('log')
ax.set_title('Total days active vs days active in Florence city, for Italians with calls in Florence')
ax.set_xlabel('Total days active')
ax.set_ylabel('Days active in Florence city')
ylim([.9,80])
xlim([.9,80])
plt.show()



