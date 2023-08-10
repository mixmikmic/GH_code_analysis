import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import *

import igraph as ig # Need to install this in your virtual environment

import psycopg2
from re import sub

import editdistance # Needs to be installed

# from pymining import seqmining

# import os
# import sys
# sys.path.append('/home/mmalik/optourism-repo' + "/pipeline")
# from firenzecard_analyzer import *

# TODO: connect with dbutils
conn_str = ""
conn = psycopg2.connect(conn_str)
cursor = conn.cursor()

nodes = pd.read_sql('select * from optourism.firenze_card_locations', con=conn)
nodes.head()

df = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
df['museum_id'].replace(to_replace=39,value=38,inplace=True)
df['short_name'] = df['museum_id'].replace(dict(zip(nodes['museum_id'],nodes['short_name'])))
df['string'] = df['museum_id'].replace(dict(zip(nodes['museum_id'],nodes['string'])))
df['date'] = pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.date
df['hour'] = pd.to_datetime(df['date']) + pd.to_timedelta(pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.hour, unit='h')
df.head()

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

df4 = df.groupby(['user_id','entry_time','date','hour','museum_name','short_name','string']).sum() # Need to group in this order to be correct further down
df4['total_people'] = df4['total_adults'] + df4['minors']
df4.head()

df4.reset_index(inplace=True)
df4.drop(['adults_first_use','adults_reuse','total_adults','minors','museum_id'], axis = 1, inplace=True)
df4.head(10)

df4.columns

df4['from'] = 'source' # Initialize 'from' column with 'source'
df4['to'] = df4['short_name'] # Copy 'to' column with row's museum_name
df4.head(10)

make_link = (df4['user_id'].shift(1)==df4['user_id'])&(df4['date'].shift(1)==df4['date']) # Row indexes at which to overwrite 'source'
df4['from'][make_link] = df4['short_name'].shift(1)[make_link]
df4.head(50)

df4['s'] = ' ' # Initialize 'from' column with 'source'
df4['t'] = df4['string'] # Copy 'to' column with row's museum_name
df4['s'][make_link] = df4['string'].shift(1)[make_link]
df4.head()

# Concatenating the source column is not enough, it leaves out the last place in the path. 
# Need to add a second 'source' column that, for the last item in a day's path, contains two characters.
df4['s2'] = df4['s']
df4['s2'][df4['from'].shift(-1)=='source'] = (df4['s2'] + df4['t'])[df4['from'].shift(-1)=='source']
# Note: the above trick doesn't work for the last row of data. So, do this as well:
df4.iloc[-1:]['s2'] = df4.iloc[-1:]['s'] + df4.iloc[-1:]['t']
df4.tail()

df4.head()

df5 = df4.groupby('user_id')['s2'].sum().to_frame() # sum() on strings concatenates 
df5.head()

df6 = df5['s2'].apply(lambda x: pd.Series(x.strip().split(' '))) # Now split along strings. Takes a few seconds.
df6.head() # Note: 4 columns is correct, Firenze card is *72 hours from first use*, not from midnight of the day of first yse!

df6.head(50) # Data stories just fall out! People traveling together, splitting off, etc. We assume this but strong coupling is hard to ignore.

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





