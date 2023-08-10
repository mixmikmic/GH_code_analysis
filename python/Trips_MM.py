import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import psycopg2

import sys
sys.path.append('../../src/')
from utils.database import dbutils

conn = dbutils.connect()
cursor = conn.cursor()

df = pd.read_sql('''
select cust_id, 
    (cust_id - lag(cust_id) over ())=0 as same_cust, 
    date_, 
    extract(days from date_ - lag(date_) over ()) - 1 as datediff,
    calls, 
    calls_in_florence_city as calls_in_fl_city,
    calls_near_airport
from optourism.foreigners_timeseries_daily;
''', con=conn)

# df = pd.read_sql('''
# select cust_id, 
#     (cust_id - lag(cust_id) over ())=0 as same_cust, 
#     date_, 
#     extract(days from date_ - lag(date_) over ()) - 1 as datediff,
#     calls, 
#     in_florence as calls_in_fl_prov, 
#     in_florence_comune as calls_in_fl_city, 
#     in_florence>0 as now_in_fl_prov, 
#     (case when (cust_id - lag(cust_id) over ())=0 then (lag(in_florence) over ())>0 else Null end) as was_in_fl_prov,
#     in_florence_comune>0 as now_in_fl_city, 
#     (case when (cust_id - lag(cust_id) over ())=0 then (lag(in_florence_comune) over ())>0 else Null end) as was_in_fl_city
# from optourism.foreigners_timeseries_daily;
# ''', con=conn)

dfi = pd.read_sql('''
select cust_id, 
    (cust_id - lag(cust_id) over ())=0 as same_cust, 
    date_, 
    extract(days from date_ - lag(date_) over ()) - 1 as datediff,
    calls, 
    calls_in_florence_city as calls_in_fl_city,
    calls_near_airport
from optourism.italians_timeseries_daily;
''', con=conn)

df.head(10) # Check

# df.iloc[0,] # How to locate an individual element

df.iloc[0,1] = False # Replace the 'None' with 'False'
df.head(20) # Check

df.loc[df['same_cust']==False,'datediff'] = None # Comes out as NaN
df.head(10)

dfi.loc[dfi['same_cust']==False,'datediff'] = None # Comes out as NaN

df['calls_out_fl_city'] = df['calls'] - df['calls_in_fl_city']
df['in_fl_city'] = df['calls_in_fl_city']>0
df['out_fl_city'] = df['calls_out_fl_city']>0

df['was_in_fl_city'] = df['in_fl_city'].shift(1)
df['was_out_fl_city'] = df['out_fl_city'].shift(1)
df['willbe_in_fl_city'] = df['in_fl_city'].shift(-1)
df['willbe_out_fl_city'] = df['out_fl_city'].shift(-1)

df.loc[df['same_cust']==False,'was_in_fl_city'] = None
df.loc[df['same_cust']==False,'was_out_fl_city'] = None

df['trip'] = ''
df.head()

# df['same_cust'][0]

df.loc[(df['same_cust']==True)&(df['datediff']<3)&(df['was_in_fl_city']==False)&(df['in_fl_city']==True),'trip'] = 'start'
df.loc[(df['same_cust']==True)&(df['datediff']<3)&(df['was_in_fl_city']==True)&(df['in_fl_city']==True),'trip'] = 'continue'
df.loc[(df['same_cust']==True)&(df['datediff']<3)&(df['was_in_fl_city']==True)&(df['in_fl_city']==True)&(df['willbe_in_fl_city']==False),'trip'] = 'end'
df.loc[(df['same_cust']==False)&((df['in_fl_city']==True)|(df['calls_near_airport']>0)),'trip'] = 'first'

df.loc[(df['same_cust']==True)&(df['same_cust'].shift(-1)==False)&((df['in_fl_city']==True)|(df['calls_near_airport']>0)),'trip'] = 'last'

df['on_trip'] = df['trip']!=''

df2 = df[['cust_id','same_cust','date_','datediff','calls_in_fl_city','calls_out_fl_city','trip','on_trip']]
df2.head()

df2['trip_id'] = (((df2['on_trip'].shift(1) != df2['on_trip']).astype(int).cumsum())*(df2['on_trip']).astype(int))

df2.loc[0:50]

dfg = df2[df2['trip_id']!=0][['cust_id','trip_id']].groupby(['cust_id','trip_id']).size().to_frame()

dfg.head(20)

dfg.groupby('cust_id').std().head()

def frequency(dataframe,columnname):
    out = dataframe[columnname].value_counts().to_frame()
    out.columns = ['frequency']
    out.index.name = columnname
    out.reset_index(inplace=True)
    out = out.sort_values(columnname)
    out['cumulative'] = out['frequency'].cumsum()/out['frequency'].sum()
    out['ccdf'] = 1 - out['cumulative']
    return out

fr_trips = frequency(dfg.groupby('cust_id').count(),0) # Distribution of lengths of gaps between trips

fr_trlen = frequency(dfg,0) # Distribution of lengths of gaps between trips

fr_dtdff = frequency(df[df['datediff']>0],'datediff') # Distribution of lengths of gaps between trips

fr_dtdff.head()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(fr_dtdff['datediff'],fr_dtdff['frequency'], linestyle='steps--')
plt.yscale('log')
plt.xscale('log')
ax.set_title('Length of gaps between trups to Florence by foreign customers')
ax.set_ylabel('Number of trips with gap of length x')
ax.set_xlabel('Gaps of trip')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(fr_trips[0],fr_trips['frequency'], linestyle='steps--')
plt.yscale('log')
plt.xscale('log')
ax.set_title('Trips to Florence by foreign customers')
ax.set_ylabel('Number of customers taking x trips')
ax.set_xlabel('Number of trips to Florence')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(fr_trlen[0],fr_trlen['frequency'], linestyle='steps--')
plt.yscale('log')
plt.xscale('log')
ax.set_title('Length of trips to Florence across foreign customers')
ax.set_ylabel('Number of trips of length x across all foreign customers')
ax.set_xlabel('Length of trip to Florence')
plt.show()

dfg[dfg.groupby('cust_id').count()==1].head(20)

fr_len1trip = frequency(dfg[dfg.groupby('cust_id').count()==1],0) # Distribution of lengths of gaps between trips

fr_len1trip.head(20)

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(fr_len1trip[0],fr_len1trip['frequency'], linestyle='steps--')
plt.yscale('log')
plt.xscale('log')
ax.set_title('Length of trip to Florence for foreign customers with 1 trip')
ax.set_ylabel('Number of trips of length x')
ax.set_xlabel('Length of trip to Florence')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=dfg.groupby('cust_id').count(),y=dfg.groupby('cust_id').mean(),s=.1)
plt.yscale('log')
plt.xscale('log')
ax.set_title('Foreigner trip length vs number of trips')
ax.set_xlabel('Number of trips')
ax.set_ylabel('Mean trip length in days')
plt.show()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=dfg.groupby('cust_id').count(),y=dfg.groupby('cust_id').sum(),s=.1)
plt.yscale('log')
plt.xscale('log')
ax.set_title('Foreigner trip length vs number of trips')
ax.set_xlabel('Number of trips')
ax.set_ylabel('Total days in Florence')
plt.show()

x = dfg.groupby('cust_id').count() + np.random.normal(loc=0,scale=25)
y = dfg.groupby('cust_id').sum() + np.random.normal(loc=0,scale=25)
f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.scatter(x=x,y=y,s=.1)
plt.yscale('log')
plt.xscale('log')
plt.xlim([.9,40])
plt.ylim([.9,150])
ax.set_title('Foreigner trip length vs number of trips')
ax.set_xlabel('Number of trips')
ax.set_ylabel('Total days in Florence')
plt.show()

dfi['calls_out_fl_city'] = dfi['calls'] - dfi['calls_in_fl_city']
dfi['in_fl_city'] = dfi['calls_in_fl_city']>0
dfi['out_fl_city'] = dfi['calls_out_fl_city']>0
dfi['was_in_fl_city'] = dfi['in_fl_city'].shift(1)
dfi['was_out_fl_city'] = dfi['out_fl_city'].shift(1)
dfi['willbe_in_fl_city'] = dfi['in_fl_city'].shift(-1)
dfi['willbe_out_fl_city'] = dfi['out_fl_city'].shift(-1)

dfi.loc[dfi['same_cust']==False,'was_in_fl_city'] = None
dfi.loc[dfi['same_cust']==False,'was_out_fl_city'] = None
dfi['trip'] = ''

dfi.loc[(dfi['same_cust']==True)&(dfi['datediff']<3)&(dfi['was_in_fl_city']==False)&(dfi['in_fl_city']==True),'trip'] = 'start'
dfi.loc[(dfi['same_cust']==True)&(dfi['datediff']<3)&(dfi['was_in_fl_city']==True)&(dfi['in_fl_city']==True),'trip'] = 'continue'
dfi.loc[(dfi['same_cust']==True)&(dfi['datediff']<3)&(dfi['was_in_fl_city']==True)&(dfi['in_fl_city']==True)&(dfi['willbe_in_fl_city']==False),'trip'] = 'end'
dfi.loc[(dfi['same_cust']==False)&((dfi['in_fl_city']==True)|(dfi['calls_near_airport']>0)),'trip'] = 'first'
dfi2 = dfi[['cust_id','same_cust','date_','datediff','calls_in_fl_city','calls_out_fl_city','trip','on_trip']]
dfi2['trip_id'] = (((dfi2['on_trip'].shift(1) != dfi2['on_trip']).astype(int).cumsum())*(dfi2['on_trip']).astype(int))
dfig = df2[df2['trip_id']!=0][['cust_id','trip_id']].groupby(['cust_id','trip_id']).size().to_frame()

fri_trips = frequency(dfig.groupby('cust_id').count(),0) # Distribution of lengths of gaps between trips
fri_trlen = frequency(dfig,0) # Distribution of lengths of gaps between trips
fri_dtdff = frequency(dfi[dfi['datediff']>0],'datediff') # Distribution of lengths of gaps between trips

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(fri_dtdff['datediff'],fri_dtdff['frequency'], linestyle='steps--')
plt.yscale('log')
plt.xscale('log')
ax.set_title('Length of gaps between trups to Florence by Italian customers')
ax.set_ylabel('Number of trips with gap of length x')
ax.set_xlabel('Gaps of trip')
plt.show()

