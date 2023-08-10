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

import os
import sys
sys.path.append('/home/mmalik/optourism-repo' + "/pipeline")
from firenzecard_analyzer import *

# TODO: connect with dbutils
conn_str = ""
conn = psycopg2.connect(conn_str)
cursor = conn.cursor()

# df = get_firenze_data(conn)

# df.head()

# ft = extract_features(df)
# ft.head()

# ft[ft['user_id']==2036595][['user_id','entry_time','total_card_use_count','day_of_week','museum_name']]

# ft.columns

# test = ft.groupby('date')['total_users_per_card'].sum()

# test.head()

# temp = df.groupby(['user_id','museum_name','entry_time']).sum()
# temp[temp['is_card_with_minors']>0].head(50)

# temp[(temp['is_card_with_minors']>0)&(temp['entry_is_adult']==0)]

nodes = pd.read_sql('select * from optourism.firenze_card_locations', con=conn)
nodes.head()

df = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
df['museum_id'].replace(to_replace=38,value=39,inplace=True)
df['short_name'] = df['museum_id'].replace(dict(zip(nodes['id'],nodes['short_name'])))
df['string'] = df['museum_id'].replace(dict(zip(nodes['id'],nodes['string'])))
df['date'] = pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.date
df['hour'] = pd.to_datetime(df['date']) + pd.to_timedelta(pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.hour, unit='h')
df.head()

# Helper function for making summary tables/distributions
def frequency(dataframe,columnname):
    out = dataframe[columnname].value_counts().to_frame()
    out.columns = ['frequency']
    out.index.name = columnname
    out.reset_index(inplace=True)
    out.sort_values(columnname,inplace=True)
    out['cumulative'] = out['frequency'].cumsum()/out['frequency'].sum()
    out['ccdf'] = 1 - out['cumulative']
    return out

timeunitname = 'hour'
timeunitcode = 'h'
df1 = df.groupby(['short_name',timeunitname]).sum()
df1['total_people'] = df1['total_adults']+df1['minors']
df1.drop(['museum_id','user_id','adults_first_use','adults_reuse','total_adults','minors'], axis=1, inplace=True)
df1.head()

df1 = df1.reindex(pd.MultiIndex.from_product([df['short_name'].unique(),pd.date_range('2016-06-01','2016-10-01',freq=timeunitcode)]), fill_value=0)
df1.reset_index(inplace=True)
df1.columns = ['short_name','hour','total_people']
df1.head()

# multiline plot with group by
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(15,8), dpi=300)
for key, grp in df1.groupby(['short_name']):
    if key in ['Accademia','Uffizi']:
        ax.plot(grp['hour'], grp['total_people'], linewidth=.5, label=str(key))
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
ax.set_xlim(['2016-06-01','2016-06-15'])
plt.show()

# multiline plot with group by
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(15,8), dpi=300)
for key, grp in df1.groupby(['short_name']):
    ax.plot(grp['hour'], grp['total_people'], linewidth=.5, label=str(key))
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
ax.set_xlim(['2016-06-01','2016-06-15'])
plt.show()

df2 = df.groupby('museum_name').sum()[['total_adults','minors']]
df2['total_people'] = df2['total_adults'] + df2['minors']
df2.sort_values('total_people',inplace=True,ascending=False)
df2.head()

df2.plot.bar(figsize=(16,8))
plt.title('Number of Firenze card visitors')
plt.xlabel('Museum')
plt.ylabel('Number of people')
# plt.yscale('log')
plt.show()

# df3 = df.sort_values(['user_id','entry_time'],ascending=False,inplace=False)
# df3.reset_index(inplace=True)
# df3.drop(['index','museum_id'], axis=1, inplace=True)
# df3.head()
# df3.groupby(['user_id','date','museum_name','entry_time']).sum().head(10) # Even though this grouping's multiindex looks nicer

df4 = df.groupby(['user_id','entry_time','date','hour','museum_name','short_name','string']).sum() # Need to group in this order to be correct further down
df4['total_people'] = df4['total_adults'] + df4['minors']
df4.head()

df4.reset_index(inplace=True)
df4.drop(['adults_first_use','adults_reuse','total_adults','minors','museum_id'], axis = 1, inplace=True)
df4.head(10)

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

df5 = df4.groupby('user_id')['s'].sum().to_frame()
df5.head()

df6 = df5['s'].apply(lambda x: pd.Series(x.strip().split(' ')))
df6.head()

df6.describe()

df6.head(50)

# df4[df4['user_id']==2016016] # Do a check: before, my incorrect groupby order caused artifacts. 

# df4[(df4['from']=="Galleria dell'Accademia di Firenze")&(df4['to']=="Galleria degli Uffizi")] # Before, this result was empty

# # This manually checked the above result, to make sure I didn't make a mistake in creating the columns
# df4[((df4['museum_name'].shift(1)=="Galleria dell'Accademia di Firenze")\
#      &(df4['museum_name']=="Galleria degli Uffizi")\
#      &(df4['user_id']==df4['user_id'].shift(1))
#      &(df4['date']==df4['date'].shift(1))
#     )\
#    | \
#     ((df4['museum_name']=="Galleria dell'Accademia di Firenze")\
#      &(df4['museum_name'].shift(-1)=="Galleria degli Uffizi")\
#      &(df4['user_id']==df4['user_id'].shift(-1))
#      &(df4['date']==df4['date'].shift(-1))
#     )]

# df4[(df4['to']=="Galleria dell'Accademia di Firenze")&(df4['from']=="Galleria degli Uffizi")] # Once the above was finished, had to make sure the opposite problem didn't happen

# Create the actual edgelist for the transition matrix (of a first-order Markov chain)
df5 = df4.groupby(['from','to'])['total_people'].sum().to_frame()
df5.columns = ['weight']
df5.reset_index(inplace=True)
df5.head(10)

# Create and check the graph
g2 = ig.Graph.TupleList(df5.itertuples(index=False), directed=True, weights=True)
ig.summary(g2)

g2.vs['name']

# Delete the dummy 'source' node
g2.delete_vertices([v.index for v in g2.vs if v['name']==u'source'])
ig.summary(g2)

# Put in graph attributes to help with plotting
g2.vs['label'] = g2.vs["name"] # [sub("'","",i.decode('unicode_escape').encode('ascii','ignore')) for i in g2.vs["name"]] # Is getting messed up!
g2.vs['size'] = [.00075*i for i in g2.strength(mode='in',weights='weight')] # .00075 is from hand-tuning

g2.vs['label']

latlon = pd.DataFrame({'short_name':g2.vs['label']}).merge(nodes[['short_name','longitude','latitude']],left_index=True,how='left',on='short_name')

g2.vs['x'] = (latlon['longitude']).values.tolist()
g2.vs['y'] = (-1*latlon['latitude']).values.tolist()
# layout = ig.Layout(latlon[['longitude','latitude']].values.tolist())
# len(layout)

# layout[0:]

# layout = g2.layout('lgl')
# len(layout)

visual_style = {}
visual_style["edge_width"] = [.001*i for i in g2.es["weight"]] # Scale weights. .001*i chosen by hand. Try also .05*np.sqrt(i)
visual_style['edge_arrow_size'] = [.00025*i for i in g2.es["weight"]] # .00025*i chosen by hand. Try also .01*np.sqrt(i)
visual_style['vertex_label_size'] = 8
visual_style['vertex_color'] = "rgba(100, 100, 255, .75)"
visual_style['edge_color'] = "rgba(0, 0, 0, .25)"
visual_style['edge_curved'] = True
ig.plot(g2.as_undirected(), 'graph.svg', bbox = (1000,1000), **visual_style)

# print(g2.get_adjacency()) # This was another check; before it was very nearly upper triangular. Now it looks much better. Copy into a text editor and resize to see the whole matrix.

transition_matrix = pd.DataFrame(g2.get_adjacency(attribute='weight').data, columns=g2.vs['name'], index=g2.vs['name'])

transition_matrix

# transition_matrix.reindex()
order = transition_matrix.sum(1).sort_values(ascending=False).to_frame().index
transition_matrix[order].reindex(order)

plt.matshow(np.log(transition_matrix[order].reindex(order)))

[order[0:1],order[0:]]

fig = plt.figure(figsize=(10,10))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.PuBu
# cax = ax.matshow(transition_matrix[order].reindex(order),cmap=cmap)
cax = ax.matshow(np.log(transition_matrix[order].reindex(order)),cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+order.tolist(),rotation=90)
ax.set_yticklabels(['']+order.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()



