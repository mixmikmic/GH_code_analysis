import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import *

import igraph as ig # Need to install this in your virtual environment
from re import sub

import editdistance # Needs to be installed
from scipy.spatial.distance import pdist, squareform
# Usage:
# editdistance.eval('banana', 'bahama')

import seaborn as sns

import sys
sys.path.append('../../src/')
from utils.database import dbutils

conn = dbutils.connect()
cursor = conn.cursor()

nodes = pd.read_sql('select * from optourism.firenze_card_locations', con=conn)

df = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
df['museum_id'].replace(to_replace=39,value=38,inplace=True)
df['short_name'] = df['museum_id'].replace(dict(zip(nodes['museum_id'],nodes['short_name'])))
df['string'] = df['museum_id'].replace(dict(zip(nodes['museum_id'],nodes['string'])))
df['date'] = pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.date
df['hour'] = pd.to_datetime(df['date']) + pd.to_timedelta(pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.hour, unit='h')
df['total_people'] = df['total_adults'] + df['minors']

# Helper function for making summary tables/distributions
def frequency(dataframe,columnname):
    out = dataframe[columnname].value_counts().to_frame()
    out.columns = ['frequency']
    out.index.name = columnname
    out.reset_index(inplace=True)
    out.sort_values('frequency',inplace=True,ascending=False)
    out['cumulative'] = out['frequency'].cumsum()/out['frequency'].sum()
    out['ccdf'] = 1 - out['cumulative']
    return out

df4 = df.groupby(['user_id','entry_time','date','hour','museum_name','short_name','string']).sum()['total_people'].to_frame() # Need to group in this order to be correct further down
df4.reset_index(inplace=True)

df4['from'] = 'source' # Initialize 'from' column with 'source'
df4['to'] = df4['short_name'] # Copy 'to' column with row's museum_name

make_link = (df4['user_id'].shift(1)==df4['user_id'])&(df4['date'].shift(1)==df4['date']) # Row indexes at which to overwrite 'source'
df4['from'][make_link] = df4['short_name'].shift(1)[make_link]

df4['s'] = ' ' # Initialize 'from' column with 'source'
df4['t'] = df4['string'] # Copy 'to' column with row's museum_name
df4['s'][make_link] = df4['string'].shift(1)[make_link]

# Concatenating the source column is not enough, it leaves out the last place in the path. 
# Need to add a second 'source' column that, for the last item in a day's path, contains two characters.
df4['s2'] = df4['s']
df4['s2'][df4['from'].shift(-1)=='source'] = (df4['s2'] + df4['t'])[df4['from'].shift(-1)=='source']
# Note: the above trick doesn't work for the last row of data. So, do this as well:
df4.iloc[-1:]['s2'] = df4.iloc[-1:]['s'] + df4.iloc[-1:]['t']

df5 = df4.groupby('user_id')['s2'].sum().to_frame() # sum() on strings concatenates 
df5.head()

df6 = df5['s2'].apply(lambda x: pd.Series(x.strip().split(' '))) # Now split along strings. Takes a few seconds.
df6.head() # Note: 4 columns is correct, Firenze card is *72 hours from first use*, not from midnight of the day of first yse!

df6.head(50) # Data stories just fall out! People traveling together, splitting off, etc. We assume this but strong coupling is hard to ignore.

fr1 = frequency(df5,'s2')

# INSIGHT: the top 15 paths are permutations of Duomo, Uffizi, Accademia. 
fr1.head(50)

fr1.iloc[0:50].plot.bar(x='s2',y='frequency',figsize=(24,10))
plt.title('Most common total Firenze card paths')
plt.xlabel('x = Encoded path')
plt.ylabel('Number of cards with total path x')
# plt.yscale('log')
plt.show()

nodes

df7 = df5['s2'].apply(lambda x: pd.Series(len(sub(' ','',x))))

df7.head()

df7.sort_values(0,ascending=False).head(10)

df6.loc[df7.sort_values(0,ascending=False).head(10).index]

fr2 = frequency(df7,0)
fr2.head()

f, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.stem(fr2[0],fr2['frequency'], linestyle='steps--')
# yscale('log')
# xscale('log')
ax.set_title('Number of museum visits by Florence Card')
ax.set_ylabel('Frequency')
ax.set_xlabel('Number of museums')
plt.show()
# NOTE: This is the number of *visits*, not people on those cards!! 
# (And, not number of museums visited, this counts multiple visits to the same museum as distinct)

df8 = df.groupby(['user_id','short_name','entry_time']).sum()['total_adults'].to_frame()
df8.head()

# Cards with more than one entrance to same museum
df9 = df.groupby(['user_id','short_name']).sum()['total_adults'].to_frame()
df9.columns = ['number_of_entries']
df9['number_of_entries'] = df9['number_of_entries']
df9[df9['number_of_entries']>1].head(50)

df8.shape[0] # Number of entries

df9.shape[0] # 12 repeat visits. Negligible.

df9[df9['number_of_entries']==1].shape[0]

df9[df9['number_of_entries']==2].shape[0]

df9[df9['number_of_entries']>2]

# # This is the number of people who entered on each card entry, not the number of repeat entries! 
# frequency(df.groupby(['user_id','short_name',]).count()['entry_time'].to_frame(),'entry_time')

df9 = df7.reset_index()
df10 = df8.reset_index()
df11 = df9.merge(df10).groupby('user_id').sum()
df11.columns = ['visits','total_people']
df11['persons_per_visit'] = df11['total_people']/df11['visits']
df11.head()

# df11[df11['persons_per_visit']>1].plot.scatter(x='visits',y='persons_per_visit')

# edit = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))

df6.fillna('',inplace=True)
df6.iloc[0:10]

def editdist(pair):
    return editdistance.eval(pair[0],pair[1])

df7 = pd.concat([df6,df6.shift()],axis=1)

df7.columns = ['0','1','2','3','0+','1+','2+','3+']
df7.head()

# df8 = df7.iloc[:,[0,4,1,5,2,6,3,7]]
# df8.columns = ['0','0+','1','1+','2','2+','3','3+']
# df8.columns = ['0','0+','1','1+','2','2+','3','3+']
# df8.head()

df7['total_edit_distance'] = df7[['0','0+']].apply(editdist,axis=1) + df7[['1','1+']].apply(editdist,axis=1) + df7[['2','2+']].apply(editdist,axis=1) + df7[['3','3+']].apply(editdist,axis=1)
df7.head()

df7['len'] = df7['0'].str.len() + df7['1'].str.len() + df7['2'].str.len() + df7['3'].str.len()
df7['len+'] = df7['0+'].str.len() + df7['1+'].str.len() + df7['2+'].str.len() + df7['3+'].str.len()
df7['len_tot'] = df7['len'] + df7['len+']
df7.head()

fr3 = frequency(df7[df7['total_edit_distance']==0],'len_tot')
fr3

frequency(df7[df7['total_edit_distance']==0],'len_tot')

df8 = df7.reset_index(inplace=False)
df8.reset_index(inplace=True)
df8.head()

# df7[df7['total_edit_distance']==0].hist('len_tot',bins=100, grid=False, figsize=[16,8])
f, ax = plt.subplots(figsize=(12,5), dpi=300)
ax.stem(fr3['len_tot']/2,fr3['frequency'], linestyle='steps--')
# yscale('log')
# xscale('log')
ax.set_title('Number of museums in perfectly matched consecutive paths')
ax.set_ylabel('Number of cards')
ax.set_xlabel('Number of museums')
plt.show()
# NOTE: This is the number of *visits*, not people on those cards!! 
# (And, not number of museums visited, this counts multiple visits to the same museum as distinct)

# df8.hist('user_id',bins=1000,figsize=[8,8])

# df8[df8['user_id']>1500000].hist('user_id',bins=1000,figsize=[8,8])

# df8.plot.scatter(x='index',y='total_edit_distance',figsize=[16,16], c=2+(df8['total_edit_distance']>0))
# sns.jointplot(x="index", y="total_edit_distance", data=df8)#, hue=(df9['total_edit_distance']==0))
# sns.jointplot(x="index", y="total_edit_distance", data=df8, kind='hex')



sns.jointplot(x="total_edit_distance", y="len_tot", data=df8)

sns.jointplot(x="total_edit_distance", y="len_tot", data=df8, kind='hex')

sns.jointplot(x="total_edit_distance", y="len_tot", data=df8, kind='kde')

df8['dist_gt_0'] = 1*(df8['total_edit_distance'] != 0)
# df8['offset'] = 1*(df8['zero_dist'] + df8['zero_dist'].shift()==0)
df8['group'] = cumsum(df8['dist_gt_0'])
df8.head(50)

df9 = df8[['group','user_id']].groupby('group').count()
df9.columns = ['people']
df9.head()

frequency(df9,'people')













# # The code below was my attempt to get a node for starting the day and ending the day from the paths. 
# # The problem is that this gives the number of _cards_, not number of people! I had to go back to the
# # dynamic edgelist construction anyway. 
# df6.head()

# df9 = df5['s2'].apply(lambda x: pd.Series(x.strip().split(' ')))
# df9.fillna(' ',inplace=True)
# df9['0_first'] = df9[0].apply(lambda x: pd.Series(x[0]))
# df9['0_last'] = df9[0].apply(lambda x: pd.Series(x[-1]))
# df9['0_len'] = df9[0].apply(lambda x: pd.Series(len(x)))
# df9['1_first'] = df9[1].apply(lambda x: pd.Series(x[0]))
# df9['1_last'] = df9[1].apply(lambda x: pd.Series(x[-1]))
# df9['1_len'] = df9[1].apply(lambda x: pd.Series(len(x)))
# df9['2_first'] = df9[2].apply(lambda x: pd.Series(x[0]))
# df9['2_last'] = df9[2].apply(lambda x: pd.Series(x[-1]))
# df9['2_len'] = df9[2].apply(lambda x: pd.Series(len(x)))
# df9['3_first'] = df9[3].apply(lambda x: pd.Series(x[0]))
# df9['3_last'] = df9[3].apply(lambda x: pd.Series(x[-1]))
# df9['3_len'] = df9[3].apply(lambda x: pd.Series(len(x)))
# df9.head()

# df9.replace(' ',np.nan,inplace=True)
# df9.head()

# from_home = frequency(df9[['0_first','1_first','2_first','3_first']].stack().to_frame(),0)[[0,'frequency']]
# from_home.columns = ['0','from_home']
# from_home.set_index('0',inplace=True)
# from_home.head()

# only = frequency(pd.concat(
#     [df9[(df9['0_len']==1)&(df9['0_first'].notnull())]['0_first'], 
#      df9[(df9['1_len']==1)&(df9['1_first'].notnull())]['1_first'], 
#      df9[(df9['2_len']==1)&(df9['2_first'].notnull())]['2_first'], 
#      df9[(df9['3_len']==1)&(df9['3_first'].notnull())]['3_first']
#     ],axis=0).to_frame()
# ,0)[[0,'frequency']]
# only.columns = ['0','only']
# only.set_index('0',inplace=True)
# only.head()

# to_home = frequency(df9[['0_last','1_last','2_last','3_last']].stack().to_frame(),0)[[0,'frequency']]
# to_home.columns = ['0','to_home']
# to_home.set_index('0',inplace=True)
# to_home.head()

# from_to_home = nodes.set_index('string')['short_name'].to_frame().join([from_home,to_home,only])
# from_to_home.set_index('short_name',inplace=True)
# from_to_home.columns = ['home_to_node','node_to_home','only_visit_of_day']
# # from_to_home['from_home'] = from_to_home['from_home_incl_only'] - from_to_home['only_visit_of_day']
# # from_to_home['to_home'] = from_to_home['to_home_incl_only'] - from_to_home['only_visit_of_day']
# from_to_home.head()

# from_to_home['home_to_node'].sort_values(ascending=False).to_frame().head(20)

# from_to_home['node_to_home'].sort_values(ascending=False).to_frame().head(20)

# from_to_home.reset_index(inplace=True)

# from_to_home

# supp_edges = pd.DataFrame({'from':['home']*from_to_home.shape[0] + from_to_home['short_name'].tolist(),
#                           'to':from_to_home['short_name'].tolist() + ['home']*from_to_home.shape[0],
#                           'weight':from_to_home['home_to_node'].tolist() + from_to_home['node_to_home'].tolist() })


# supp_edges.dropna(how='any',inplace=True)
# supp_edges



















frequency(df6,0).head()

frequency(df6,1).head()

frequency(df6,2).head()

frequency(df6,3).head()

pt = pd.concat([frequency(df6,0),frequency(df6,1),frequency(df6,2),frequency(df6,3)])
pt['daily_path'] = pt[0].replace(np.nan, '', regex=True) + pt[1].replace(np.nan, '', regex=True) + pt[2].replace(np.nan, '', regex=True) + pt[3].replace(np.nan, '', regex=True)
pt.drop([0,1,2,3,'ccdf','cumulative'],axis=1,inplace=True)
pt.head()

pt2 = pt.groupby('daily_path').sum()
pt2.sort_values('frequency', inplace=True, ascending=False)
pt2.head()

pt2[pt2['frequency']>200].plot.bar(figsize=(16,8))
plt.title('Most common daily Firenze card paths across all days')
plt.xlabel('x = Encoded path')
plt.ylabel('Number of cards with daily path x')
# plt.yscale('log')
plt.show()

nodes.head()

# For reference, here are the displayed museums
# nodes[['string','short_name']].set_index('string').reindex(['D','P','U','A','V','T','N','C','G','B','S','c','m','M','b','Y','2'])
nodes[nodes['string'].isin(['D','P','U','A','V','T','N','C','G','B','S','c','m','M','b','Y','2'])][['string','short_name']]

df6[pd.isnull(df6[0].str[0])].head()

df6.to_csv('encoded_paths.csv')

nodes.to_csv('encoded_paths_legend.csv')



df6.values





