import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Load data
uda = pd.read_csv("./aws-data/user_dist.txt", sep="\t") # User distribution, all
udf = pd.read_csv("./aws-data/user_dist_fl.txt", sep="\t") # User distribution, Florence

dra = pd.read_csv("./aws-data/user_duration.txt", sep="\t") # Duration, all
drf = pd.read_csv("./aws-data/user_duration_fl.txt", sep="\t") # Duration, Florence

dra['min'] = pd.to_datetime(dra['min'], format='%Y-%m-%d%H:%M:%S')
dra['max'] = pd.to_datetime(dra['max'], format='%Y-%m-%d%H:%M:%S')
drf['min'] = pd.to_datetime(drf['min'], format='%Y-%m-%d%H:%M:%S')
drf['max'] = pd.to_datetime(drf['max'], format='%Y-%m-%d%H:%M:%S')

dra['duration'] = dra['max'] - dra['min']
drf['duration'] = drf['max'] - drf['min']

dra['days'] = dra['duration'].dt.days
drf['days'] = drf['duration'].dt.days

cda = pd.read_csv("./aws-data/calls_per_day.txt", sep="\t") # Calls per day, all
cdf = pd.read_csv("./aws-data/calls_per_day_fl.txt", sep="\t") # Calls per day, Florence
cda['day_'] = pd.to_datetime(cda['day_'], format='%Y-%m-%d%H:%M:%S').dt.date
cdf['day_'] = pd.to_datetime(cdf['day_'], format='%Y-%m-%d%H:%M:%S').dt.date

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

data_italians = pd.read_csv("./aws-data/firence_italians_3days_past_future_sample_1K_custs.csv", header=None)
data_italians.columns = ['lat', 'lon', 'date_time_m', 'home_region', 'cust_id', 'in_florence']
regions = np.array(data_italians['home_region'].unique())
regions

'Sardegna' in data['home_region']

