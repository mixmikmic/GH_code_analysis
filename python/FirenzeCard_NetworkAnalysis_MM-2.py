import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pylab import *

import igraph as ig # Need to install this in your virtual environment

from re import sub

import sys
sys.path.append('../../src/')
from utils.database import dbutils

conn = dbutils.connect()
cursor = conn.cursor()

df = pd.read_sql('select * from optourism.firenze_card_logs', con=conn)
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

# Make a two-mode weighted edgelist
df1 = df.groupby(['user_id','museum_name'])['total_adults','minors'].sum()
df1['total_visitors'] = df1['total_adults'] + df1['minors']
df1.reset_index(inplace=True)
df1.drop(['total_adults','minors'], axis=1, inplace=True)
df1.head()

g = ig.Graph.TupleList(df1.itertuples(index=False), weights=True) # TupleList is how to get from pandas to igraph

ig.summary(g) # check to make sure the graph imported successfully

g.get_edgelist()[0:20] # There is no longer a head() method, so we have to use usual indexing. 

g.vs["name"][0:10] # How to see node names. "vs" stands for "vertices"

g.get_edgelist()[1:10] # How to see edges. They have internal IDs, but for us are referenced as a unique tuple. 

g.es["weight"][0:25] # How to see edge properties

# This network has two types of nodes: user_ids, and museums. 
# Python igraph doesn't automatically recognize/track different node types, but
#  fortunately, their names mean we can easily tell them apart: user_ids are 
#  numbers, and museums are not. 
# We associate a "type" attribute with each node, and can use this for 
#  igraph methods for bipartite/two-mode networks. 
g.vs["type"] = [isinstance(name,int)==False for name in g.vs["name"]]

# # This is how to do it taking it into Pandas then coming back out again
# s = pd.Series(g.vs["name"]) # Turn the list into a pandas series
# print s.head()
# print (s.str.isnumeric()==False).astype('int').tolist()[0:9] # Perform an element-wise operation, then take back to a list
# g.vs["type"] = (s.str.isnumeric()==False).tolist()

g.vs["type"][0:10]

# This turns the affiliation matrix with two types of nodes into a similarity
#  matrix between one of those two types. Similarity here is sharing ties to 
#  nodes of the same type. The user-user similarity matrix is too big to compute,
#  so we only get one of the projections. The output is an undirected, weighted
#  network, where the weights are the number of shared connections to nodes of 
#  the other type. 
g_m = g.bipartite_projection(which=True) 

ig.summary(g_m)

# print(g_m) # Gives "adjacency list"

g_m.get_edgelist()[0:10]

g_m.es["weight"][0:10] # These weights represent the number of user_ids in common

# # To do visualizations, run these in your virtual environment
# pip install cffi
# pip install cairocffi

g_m.vs["label"] = g_m.vs["name"] # "label" attribute is used by igraph automatically for naming nodes on plots

ig.plot(g_m, bbox = (700,500), layout = g_m.layout('kk')) # "kk" is Kamada-Kawai, a standard layout algorithm
# Note that Kamada-Kawai is stochastic, so multiple runs will 
#  generate slightly different graphs (the main difference is 
#  orientation, but the shape differs slightly as well)

fr_ew = frequency(pd.Series(g_m.es["weight"]).to_frame(),0)
fr_ew.head(20)

pd.Series(g_m.es["weight"]).to_frame().plot.hist(y=0, logy=True, figsize=(10,8), bins=50)
plt.ylabel('Counts')
plt.xlabel('Edge weight')
plt.title('Histogram of number of shared visitors')
plt.axvline(1000,color="black") # I decided to use 1000. A round number, cuts off the peak of the histogram, and works well below. 
# plt.savefig('shared_visitors.png')
plt.show()

# This is messier but gives more justification to using 1000; that's before there are uniquely large edge weights.
# Maybe it should be a bit larger than 1000, but it's the nearest round number. 
f, ax = plt.subplots(figsize=(10,8)) #, dpi=300)
ax.stem(fr_ew[0],fr_ew['frequency'], linestyle='steps--')
yscale('log')
xscale('log')
ax.set_title('Histogram of number of shared visitors')
ax.set_ylabel('Counts')
ax.set_xlabel('Edge weight')
plt.axvline(1000,color="black")
plt.show()

# # CDF plot. Not helpful.
# f, ax = plt.subplots(figsize=(10,8)) #, dpi=300)
# ax.plot(fr_ew[0],fr_ew['cumulative'])
# # yscale('log')
# # xscale('log')
# ax.set_title('Shared visitors')
# ax.set_ylabel('Fraction of edges with weight x or less')
# ax.set_xlabel('Weight')
# plt.show()

# # CCDF/Survival function plot. Not helpful.
# f, ax = plt.subplots(figsize=(10,8)) #, dpi=300)
# ax.plot(fr_ew[0],fr_ew['ccdf'])
# # yscale('log')
# # xscale('log')
# ax.set_title('Shared visitors')
# ax.set_ylabel('Fraction of edges with weight x or greater')
# ax.set_xlabel('Weight')
# plt.show()

ig.summary(g_m) # How many edges are there initially?

g_m.es.select(weight_lt=1000).delete() # Deletes edges with weights under 1000. Modifies graph object in place. 
ig.summary(g_m) # See the result. 798 edges to 194. 

visual_style = {}
visual_style["edge_width"] = [.0001*i for i in g_m.es["weight"]] # Scale weights
ig.plot(g_m, bbox = (700,1000), layout = g_m.layout('kk'), **visual_style)

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

df['date'] = pd.to_datetime(df['entry_time'], format='%Y-%m-%d %H:%M:%S').dt.date
df.head(10)

df3 = df.sort_values(['user_id','entry_time'],ascending=False,inplace=False)
df3.reset_index(inplace=True)
df3.drop(['index','museum_id'], axis=1, inplace=True)
df3.head()

df4 = df3.groupby(['user_id','entry_time','date','museum_name']).sum() # Need to group in this order to be correct further down
df4['total_people'] = df4['total_adults'] + df4['minors']
df4.head()

df3.groupby(['user_id','date','museum_name','entry_time']).sum().head(10) # Even though this grouping's multiindex looks nicer

df4.reset_index(inplace=True)
df4.head(10)

df4['from'] = u'source' # Initialize 'from' column with 'source'
df4['to'] = df4['museum_name'] # Copy 'to' column with row's museum_name

df4.head(10)

make_link = (df4['user_id'].shift(1)==df4['user_id'])&(df4['date'].shift(1)==df4['date']) # Row indexes at which to overwrite 'source'
df4['from'][make_link] = df4['museum_name'].shift(1)[make_link]
df4.head(50)

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

# Put in graph attributes to help with plotting
g2.vs['label'] = g2.vs["name"] # [sub("'","",i.decode('unicode_escape').encode('ascii','ignore')) for i in g2.vs["name"]] # Is getting messed up!
g2.vs['size'] = [.00075*i for i in g2.strength(mode='in',weights='weight')] # .00075 is from hand-tuning

g2.vs['label']

layout = g2.layout('lgl')

visual_style = {}
visual_style["edge_width"] = [.001*i for i in g2.es["weight"]] # Scale weights. .001*i chosen by hand. Try also .05*np.sqrt(i)
visual_style['edge_arrow_size'] = [.00025*i for i in g2.es["weight"]] # .00025*i chosen by hand. Try also .01*np.sqrt(i)
visual_style['vertex_label_size'] = 8
visual_style['vertex_color'] = "rgba(100, 100, 255, .75)"
visual_style['edge_color'] = "rgba(0, 0, 0, .25)"
visual_style['edge_curved'] = True
# ig.plot(g2, bbox = (700,1000), layout = layout, margin=20, **visual_style)
ig.plot(g2, 'graph.svg', bbox = (1000,1000), **visual_style)

# print(g2.get_adjacency()) # This was another check; before it was very nearly upper triangular. Now it looks much better. Copy into a text editor and resize to see the whole matrix.

transition_matrix = pd.DataFrame(g2.get_adjacency(attribute='weight').data, columns=g2.vs['name'], index=g2.vs['name'])

plt.matshow(np.log(transition_matrix))



