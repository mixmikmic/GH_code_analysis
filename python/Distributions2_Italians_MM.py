import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pylab import *

import sys
sys.path.append('../../src/')
from utils.database import dbutils

conn = dbutils.connect()
cursor = conn.cursor()

df = pd.read_sql('select * from optourism.italians_counts', con=conn)
df.head()

# 1.1, 1.2, 1.3
frc_ua = df['calls'].value_counts().to_frame()
frc_ua.columns = ['frequency']
frc_ua.index.name = 'calls'
frc_ua.reset_index(inplace=True)
frc_ua = frc_ua.sort_values('calls')
frc_ua['cumulative'] = frc_ua['frequency'].cumsum()/frc_ua['frequency'].sum()
frc_ua['ccdf'] = 1 - frc_ua['cumulative']

frc_uc = df['calls_in_florence_comune'].value_counts().to_frame()
frc_uc.columns = ['frequency']
frc_uc.index.name = 'calls_in_florence_comune'
frc_uc.reset_index(inplace=True)
frc_uc = frc_uc.sort_values('calls_in_florence_comune')
frc_uc['cumulative'] = frc_uc['frequency'].cumsum()/frc_uc['frequency'].sum()
frc_uc['ccdf'] = 1 - frc_uc['cumulative']

# 2.1, 2.2, 2.3
frc_da = df['days_active'].value_counts().to_frame()
frc_da.columns = ['frequency']
frc_da.index.name = 'days_active'
frc_da.reset_index(inplace=True)
frc_da = frc_da.sort_values('days_active')
frc_da['cumulative'] = frc_da['frequency'].cumsum()/frc_da['frequency'].sum()
frc_da['ccdf'] = 1 - frc_da['cumulative']

# frc_dc = dfc['days_active_in_florence_comune'].value_counts().to_frame()
# frc_dc.columns = ['frequency']
# frc_dc.index.name = 'days_active_in_florence_comune'
# frc_dc.reset_index(inplace=True)
# frc_dc = frc_dc.sort_values('days_active_in_florence_comune')
# frc_dc['cumulative'] = frc_dc['frequency'].cumsum()/frc_dc['frequency'].sum()
# frc_dc['ccdf'] = 1 - frc_dc['cumulative']

frc2 = df[df['calls_in_florence_comune']>0]['calls'].value_counts().to_frame()
frc2.columns = ['frequency']
frc2.index.name = 'calls'
frc2.reset_index(inplace=True)
frc2 = frc2.sort_values('calls')
frc2['cumulative'] = frc2['frequency'].cumsum()/frc2['frequency'].sum()
frc2['ccdf'] = 1 - frc2['cumulative']

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(frc2['calls'],frc2['frequency'], linestyle='steps--')
yscale('log')
xscale('log')
ax.set_title('Calls per person among Italians who have made calls in Florence city')
ax.set_ylabel('Number of people making x calls')
ax.set_xlabel('Number of calls')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.plot(frc2['calls'],frc2['cumulative'])
# yscale('log')
xscale('log')
# ylim([.7,1.01])
ax.set_title('CDF of calls per person among Italians who have made calls in Florence city')
ax.set_ylabel('Proportion of users making x or fewer calls')
ax.set_xlabel('Number of calls')
axvline(4.5) # Our cutoff
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
ylim([.7,1.01])
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

# f, ax = plt.subplots(figsize=(6,5), dpi=300)
# ax.stem(frc_dc['days_active_in_florence_comune'],frc_dc['frequency'], linestyle='steps--')
# yscale('log')
# xscale('log')
# ax.set_title('Days active per person among Italians in Florence city')
# ax.set_ylabel('Number of people with x days active')
# ax.set_xlabel('Days active')
# plt.show()

# f, ax = plt.subplots(figsize=(6,5), dpi=300)
# ax.plot(frc_dc['days_active_in_florence_comune'],frc_dc['cumulative'])
# # yscale('log')
# xscale('log')
# ylim([.83,1.01])
# ax.set_title('CDF of days active per person among Italians in Florence city')
# ax.set_ylabel('Proportion of users active on x or fewer days')
# ax.set_xlabel('Number of days active')
# plt.show()

# f, ax = plt.subplots(figsize=(6,5), dpi=300)
# ax.plot(frc_dc['days_active_in_florence_comune'],frc_dc['ccdf'])
# yscale('log')
# xscale('log')
# ax.set_title('CCDF of days active per person among Italians in Florence city')
# ax.set_ylabel('Proportion of users active on at least x days')
# ax.set_xlabel('Number of days a')
# plt.show()

df['mean_calls_per_day'] = df['calls']/df['days_active']
df[df['calls_in_florence_comune']>0].head()

print df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].max()
print df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].min()
print df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].mean()
print df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].median()
print df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].std()
print df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].std()*2+df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].mean()
print df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].std()*3+df[df['calls_in_florence_comune']>0]['mean_calls_per_day'].mean()

df[(df['calls_in_florence_comune']>0)&(df['mean_calls_per_day']<1000)].plot.hist(y='mean_calls_per_day', logy=True, figsize=(15,10), bins=200)
plt.ylabel('Frequency')
plt.xlabel('Average calls per active day')
plt.axvline(150,color="black")
# plt.xlim([0,1000])
plt.title('Average calls per active day by Italian SIM cards who were in Florence city')

# dfc.plot.scatter(x='calls',y='days_active',figsize=(15,10),logy=True,logx=True)

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=df['calls'],y=df['days_active'],s=.1)
yscale('log')
xscale('log')
ax.set_title('Calls by days active among Italians')
ax.set_xlabel('Calls')
ax.set_ylabel('Days active')
# x=[200, 200*200]
y=[1/10000, 1*200]
x2=[200/10000, 200*200]
# x3=[100, 100*400]
ylim([1,150])
# plt.plot(x,y,color='black',linewidth=.5)
plt.plot(x2,y,color='black',linewidth=.5)
# plt.plot(x3,y,color='black',linewidth=.5)
ax.axvline(4.5,color='black',linewidth=.5)
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=df['calls'],y=df['calls_in_florence_comune'],s=.1)
yscale('log')
xscale('log')
ax.set_title('Total calls vs calls in Florence city, for Italians with calls in Florence')
ax.set_xlabel('Total calls')
ax.set_ylabel('Calls in Florence city')
xlim([.9,10*10000])
ylim([.9,10*10000])
plt.show()

# f, ax = plt.subplots(figsize=(6,5), dpi=300)
# ax.scatter(x=df['days_active'],y=df['days_active_in_florence_comune'],s=.1)
# yscale('log')
# xscale('log')
# ax.set_title('Total days active vs days active in Florence city, for Italians with calls in Florence')
# ax.set_xlabel('Total days active')
# ax.set_ylabel('Days active in Florence city')
# xlim([.9,10*100])
# ylim([.9,10*100])
# plt.show()

df3 = pd.read_csv("/mnt/data/shared/aws-data/restricted-data/CDR-data/cust_foreigners_timeseries_GEN.csv")
df3.head()

frc3 = df3.groupby(['cust_id'])['in_florence_comune'].max()

frc3.plot.hist(logy=True, figsize=(15,10), bins=2000)
plt.ylabel('Frequency')
plt.xlabel('Maximum daily calls in Florence city')
plt.axvline(4.5,color="black")
plt.xlim([0,300])
plt.title('Maximum daily calls in Florence city among foreigners')


# frc3.columns = ['frequency']
# frc3.index.name = 'calls'
# frc3.reset_index(inplace=True)
# frc3 = frc2.sort_values('calls')
# frc3['cumulative'] = frc2['frequency'].cumsum()/frc2['frequency'].sum()
# frc3['ccdf'] = 1 - frc2['cumulative']
frc3.plot.hist(bins=100)

f, ax = plt.subplots(figsize=(15,7.5), dpi=180)
ax.stem(frt_dc['days'],frt_dc['frequency'], linestyle='steps--')
# yscale('log')
# xscale('log')
ax.set_title('Histogram of number of towers in Florence city with x calls')
ax.set_ylabel('Frequency')
ax.set_xlabel('Number of calls')
plt.show()

f, ax = plt.subplots(figsize=(15,7.5), dpi=180)
ax.stem(frt_c['calls'],frt_c['frequency'], linestyle='steps--')
yscale('log')
xscale('log')
ax.set_title('Histogram of number of towers in Florence city with x calls')
ax.set_ylabel('Frequency')
ax.set_xlabel('Number of calls')
plt.show()

'Sardegna' in data['home_region']

data_italians = pd.read_csv("./aws-data/firence_italians_3days_past_future_sample_1K_custs.csv", header=None)
data_italians.columns = ['lat', 'lon', 'date_time_m', 'home_region', 'cust_id', 'in_florence']
regions = np.array(data_italians['home_region'].unique())
regions

# x = np.linspace(0.1, 2*np.pi, 10)
# x

# np.cos(x)

# x = frc_uc['calls_in_florence_comune'].as_matrix()
# x





cda.head()

mcpdf = cdf.groupby('cust_id')['count'].mean().to_frame() # Mean calls per day, Florence
mcpdf.columns = ['mean_calls_per_day']
mcpdf = mcpdf.sort_values('mean_calls_per_day',ascending=False)
mcpdf.index.name = 'cust_id'
mcpdf.reset_index(inplace=True)
mcpdf.head()

# mcpdf.plot(y='mean_calls_per_day', style='.', logy=True, figsize=(10,10))
mcpdf.plot.hist(y='mean_calls_per_day', logy=True, figsize=(10,10), bins=100)
plt.ylabel('Number of customers with x average calls per day')
# plt.xlabel('Customer rank')
plt.title('Mean number of calls per day during days in Florence by foreign SIM cards')

cvd = udf.merge(drf, left_on='cust_id', right_on='cust_id', how='outer') # Count versus days
cvd.plot.scatter(x='days', y='count', s=.1, figsize = (10, 10))
plt.ylabel('Number of calls')
plt.xlabel('Duration between first and last days active')
plt.title('Calls versus duration of records of foreign SIMs in Florence')

fr = drf['days'].value_counts().to_frame() # NOTE: FIGURE OUT HOW TO ROUND, NOT TRUNCATE
fr.columns = ['frequency']
fr.index.name = 'days'
fr.reset_index(inplace=True)
fr = fr.sort_values('days')
fr['cumulative'] = fr['frequency'].cumsum()/fr['frequency'].sum()

fr.plot(x='days', y='frequency', style='o-', logy=True, figsize = (10, 10))
plt.ylabel('Number of people')
plt.axvline(14,ls='dotted')
plt.title('Foreign SIM days between first and last instances in Florence')

cvd = udf.merge(drf, left_on='cust_id', right_on='cust_id', how='outer') # Count versus days
cvd.plot.scatter(x='days', y='count', s=.1, figsize = (10, 10))
plt.ylabel('Number of calls')
plt.xlabel('Duration between first and last days active')
plt.title('Calls versus duration of records of foreign SIMs in Florence')



fr = udf['count'].value_counts().to_frame()
fr.columns = ['frequency']
fr.index.name = 'calls'
fr.reset_index(inplace=True)
fr = fr.sort_values('calls')
fr['cumulative'] = fr['frequency'].cumsum()/fr['frequency'].sum()
fr.head()

fr.plot(x='calls', y='frequency', style='o-', logx=True, figsize = (10, 10))
# plt.axvline(5,ls='dotted')
plt.ylabel('Number of people')
plt.title('Number of people placing or receiving x number of calls over 4 months')

fr.plot(x='calls', y='cumulative', style='o-', logx=True, figsize = (10, 10))
plt.axhline(1.0,ls='dotted',lw=.5)
plt.axhline(.90,ls='dotted',lw=.5)
plt.axhline(.75,ls='dotted',lw=.5)
plt.axhline(.67,ls='dotted',lw=.5)
plt.axhline(.50,ls='dotted',lw=.5)
plt.axhline(.33,ls='dotted',lw=.5)
plt.axhline(.25,ls='dotted',lw=.5)
plt.axhline(.10,ls='dotted',lw=.5)
plt.axhline(0.0,ls='dotted',lw=.5)
plt.axvline(max(fr['calls'][fr['cumulative']<.90]),ls='dotted',lw=.5)
plt.ylabel('Cumulative fraction of people')
plt.title('Cumulative fraction of people placing or receiving x number of calls over 4 months')

df2 = pd.read_table("./aws-data/towers_with_counts2.txt")
df2.head()

fr2 = df2['count'].value_counts().to_frame()
fr2.columns = ['frequency']
fr2.index.name = 'count'
fr2.reset_index(inplace=True)
fr2 = fr2.sort_values('count')
fr2['cumulative'] = fr2['frequency'].cumsum()/fr2['frequency'].sum()
fr2.head()

fr2.plot(x='count', y='frequency', style='o-', logx=True, figsize = (10, 10))
# plt.axvline(5,ls='dotted')
plt.ylabel('Number of cell towers')
plt.title('Number of towers with x number of calls placed or received over 4 months')

fr2.plot(x='count', y='cumulative', style='o-', logx=True, figsize = (10, 10))
plt.axhline(0.1,ls='dotted',lw=.5)
plt.axvline(max(fr2['count'][fr2['cumulative']<.10]),ls='dotted',lw=.5)
plt.axhline(0.5,ls='dotted',lw=.5)
plt.axvline(max(fr2['count'][fr2['cumulative']<.50]),ls='dotted',lw=.5)
plt.axhline(0.9,ls='dotted',lw=.5)
plt.axvline(max(fr2['count'][fr2['cumulative']<.90]),ls='dotted',lw=.5)
plt.ylabel('Cumulative fraction of cell towers')
plt.title('Cumulative fraction of towers with x number of calls  placed or received over 4 months')

df['datetime'] = pd.to_datetime(df['date_time_m'], format='%Y-%m-%d %H:%M:%S')
df['date'] = df['datetime'].dt.floor('d') # Faster than df['datetime'].dt.date

df2 = df.groupby(['cust_id','date']).size().to_frame()
df2.columns = ['count']
df2.index.name = 'date'
df2.reset_index(inplace=True)
df2.head(20)

df3 = (df2.groupby('cust_id')['date'].max() - df2.groupby('cust_id')['date'].min()).to_frame()
df3['calls'] = df2.groupby('cust_id')['count'].sum()
df3.columns = ['days','calls']
df3['days'] = df3['days'].dt.days
df3.head()

fr = df['cust_id'].value_counts().to_frame()['cust_id'].value_counts().to_frame()

# plt.scatter(np.log(df3['days']), np.log(df3['calls']))
# plt.show()





fr.plot(x='calls', y='freq', style='o', logx=True, logy=True)

x=np.log(fr['calls'])
y=np.log(1-fr['freq'].cumsum()/fr['freq'].sum())
plt.plot(x, y, 'r-')

# How many home_Regions
np.count_nonzero(data['home_region'].unique())

# How many customers
np.count_nonzero(data['cust_id'].unique())

# How many Nulls are there in the customer ID column?
df['cust_id'].isnull().sum()

# How many missing data are there in the customer ID?
len(df['cust_id']) - df['cust_id'].count()

df['cust_id'].unique()

