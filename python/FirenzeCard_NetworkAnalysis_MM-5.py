import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from pylab import *

import igraph as ig # Need to install this in your virtual environment

import psycopg2
# from re import sub

from scipy.spatial.distance import squareform, pdist # Use for getting weights for layout

from scipy.cluster.hierarchy import dendrogram, linkage

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
df['museum_id'].replace(to_replace=38,value=39,inplace=True)
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
    out.sort_values(columnname,inplace=True)
    out['cumulative'] = out['frequency'].cumsum()/out['frequency'].sum()
    out['ccdf'] = 1 - out['cumulative']
    return out

timeunitname = 'hour' # 'day'
timeunitcode = 'h' # 'D'
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
    if key in ['Accademia','Uffizi','Opera del Duomo']:
        ax.plot(grp['hour'], grp['total_people'], linewidth=.5, label=str(key))
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
ax.set_xlim(['2016-06-03','2016-06-15'])
ax.set_ylim([-1,110])
# plt.xticks(pd.date_range('2016-06-01','2016-06-15',freq='D'))
ticks = pd.date_range('2016-06-03','2016-06-15',freq='D').date
plt.xticks(ticks, ticks, rotation='vertical')
ax.set_xticks(pd.date_range('2016-06-03','2016-06-15',freq='6h'), minor=True, )
# fig.autofmt_xdate()
# ax.fmt_xdata = mdates.DateFormatter('%m-%d')
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
df4.head(10)

# Create the actual edgelist for the transition matrix (of a first-order Markov chain)
df5 = df4.groupby(['from','to'])['total_people'].sum().to_frame()
df5.columns = ['weight']
df5.reset_index(inplace=True)
df5.head(10)

# Make exportable dynamic edgelist
# df4[['from','to','total_people','entry_time']].sort_values('entry_time').to_csv('dynamic_edgelist.csv')
out = df4[['from','to','total_people','entry_time']].sort_values('total_people',ascending=False)
out.columns = ['source','target','people','datetime']
out.to_csv('dynamic_edgelist.csv',index=False)

# # Make actual graph object to export as gml
# out = df4[['from','to','total_people','entry_time']].sort_values('total_people',ascending=False)
# out.columns = ['source','target','people','datetime']

# g = ig.Graph.TupleList(df4.itertuples(index=False), directed=True, weights=True)
# ig.summary(g)

# g.es.attributes()

# Create and check the graph
g = ig.Graph.TupleList(df5.itertuples(index=False), directed=True, weights=True)
ig.summary(g)

# g.vs['name']

# Save the weighted indegree calculated with the source node before dropping it
indeg = g.strength(mode='in',weights='weight')[0:-1] # This drops the "source" node, which is last

# Delete the dummy 'source' node
g.delete_vertices([v.index for v in g.vs if v['name']==u'source'])
g.simplify(loops=False, combine_edges=sum)
ig.summary(g)

# Put in graph attributes to help with plotting
g.vs['label'] = g.vs["name"] 
# g.vs[sub("'","",i.decode('unicode_escape').encode('ascii','ignore')) for i in g2.vs["name"]] # Is getting messed up!

# g2.vs['label']

# Get coordinates, requires this lengthy query
latlon = pd.DataFrame({'short_name':g.vs['label']}).merge(nodes[['short_name','longitude','latitude']],left_index=True,how='left',on='short_name')

# Latitude is flipped, need to multiply by -1 to get correct orientation
g.vs['x'] = (latlon['longitude']).values.tolist()
g.vs['y'] = (-1*latlon['latitude']).values.tolist()

# # Make distances matrix for a layout based on geography but with enough spacing to not overplot and cause ZeroDivisionError
# # Want to convert the distances into attraction, do with "gravity" of 1/(d^2)
# dist = pd.DataFrame(squareform(pdist(nodes.iloc[:, 1:3])), columns=nodes['short_name'], index=nodes['short_name'])
# grav = dist.pow(-2)
# np.fill_diagonal(grav.values, 0)
# grav.head()

# A = grav.values
# g2 = ig.Graph.Adjacency((A > 0).tolist())
# g2.es['weight'] = A[A.nonzero()]
# g2.vs['label'] = grav.index

# layout = g2.layout("fr",weights='weight')
# ig.plot(g2,layout=layout)

# layout = g.layout_fruchterman_reingold(seed=nodes[['longitude','latitude']].values.tolist(),maxiter=5,maxdelta=.01)
# ig.plot(g,layout=layout)

g.delete_edges(g.es.find(_between=(g.vs(name_eq='Torre di Palazzo Vecchio'), g.vs(name_eq='M. Palazzo Vecchio'))))
g.delete_edges(g.es.find(_between=(g.vs(name_eq='M. Palazzo Vecchio'),g.vs(name_eq='Torre di Palazzo Vecchio'))))
ig.summary(g)

visual_style = {}
visual_style['vertex_size'] = [.000075*i for i in indeg] # .00075 is from hand-tuning
visual_style['vertex_label_size'] = [.00025*i for i in indeg]
visual_style['edge_width'] = [np.floor(.001*i) for i in g.es["weight"]] # Scale weights. .001*i chosen by hand. Try also .05*np.sqrt(i)
# visual_style['edge_curved'] = True
# visual_style["autocurve"] = True
ig.plot(g.as_undirected(), **visual_style) # Positions, for reference

# layout = g.layout_fruchterman_reingold(seed=nodes[['longitude','latitude']].values.tolist(),maxiter=4)
# layout = g.layout_drl(seed=nodes[['longitude','latitude']].values.tolist(), fixed=[True]*len(g.vs))
# layout = g.layout_drl(seed=nodes[['longitude','latitude']].values.tolist(), fixed=[True]*len(g.vs), weights='weight')
# layout = g.layout_graphopt(seed=nodes[['longitude','latitude']].values.tolist(), niter=1, node_mass=1)
visual_style = {}
visual_style['edge_width'] = [np.floor(.001*i) for i in g.es["weight"]] # Scale weights. .001*i chosen by hand. Try also .05*np.sqrt(i)
visual_style['edge_arrow_size'] = [.00025*i for i in g.es["weight"]] # .00025*i chosen by hand. Try also .01*np.sqrt(i)
visual_style['vertex_size'] = [.00075*i for i in indeg] # .00075 is from hand-tuning
visual_style['vertex_label_size'] = [.00025*i for i in indeg]
visual_style['vertex_color'] = "rgba(100, 100, 255, .75)"
visual_style['edge_color'] = "rgba(0, 0, 0, .25)"
# visual_style['edge_curved'] = True
visual_style["autocurve"] = True
ig.plot(g, 'graph.svg', bbox = (1000,1000), **visual_style)

# print(g2.get_adjacency()) # This was another check; before it was very nearly upper triangular. Now it looks much better. Copy into a text editor and resize to see the whole matrix.

transition_matrix = pd.DataFrame(g.get_adjacency(attribute='weight').data, columns=g.vs['name'], index=g.vs['name'])

# transition_matrix.loc[transition_matrix.idxmax(axis=0).values].idxmax(axis=1).index

# pd.Index(pd.Series(transition_matrix.loc[transition_matrix.idxmax(axis=0).values].max(axis=1).sort_values(ascending=False).index.unique().tolist() + transition_matrix.max(axis=1).sort_values(ascending=False).to_frame().index.tolist()).unique())

# temp = pd.DataFrame({'museum':transition_matrix.idxmax(axis=0).values,'values':transition_matrix.max(axis=0).values}).groupby('museum').max().sort_values('values',ascending=False)
# temp.reset_index(inplace=True)
# temp2 = transition_matrix.max(axis=0).to_frame().reset_index()
# temp2.columns = ['museum','values']
# temp = temp.append(temp2)
# order = temp.groupby('museum').max().sort_values('values',ascending=False).index

# order

# order = pd.Index([u'Opera del Duomo', 
#                   u'Accademia', 
#                   u'Uffizi', 
#                   u'Palazzo Pitti', 
#                   u'M. Palazzo Vecchio', 
#                   u'M. Galileo', 
#                   u'Santa Croce', 
#                   u'Cappelle Medicee', 
#                   u'San Lorenzo', 
#                   u'Laurenziana', 
#                   u'Palazzo Medici', 
#                   u'M. Bargello', 
#                   u'M. Santa Maria Novella', 
#                   u'M. San Marco', 
#                   u'Torre di Palazzo Vecchio', 
#                   u'M. Casa Dante', 
#                   u'Brancacci', 
#                   u'V. Bardini', 
#                   u'M. Archeologico', 
#                   u'Casa Buonarroti', 
#                   u'M. Innocenti', 
#                   u'M. Novecento', 
#                   u'La Specola', 
#                   u'Palazzo Strozzi', 
#                   u'M. Opificio', 
#                   u'Orto Botanico', 
#                   u'M. Geologia', 
#                   u'M. Mineralogia', 
#                   u'M. Ebraico', 
#                   u'M. Antropologia', 
#                   u'M. Ferragamo', 
#                   u'M. Palazzo Davanzati', 
#                   u'M. Horne', 
#                   u'M. Civici Fiesole', 
#                   u'M. Marini', 
#                   u'M. Stefano Bardini', 
#                   u'Planetario', 
#                   u'M. Stibbert', 
#                   u'M. Calcio', 
#                   u'M. Preistoria', 
#                   u'Primo Conti'])

# transition_matrix.reindex()
# transition_matrix.idxmax(axis=0).values is like "which.max()" in R: which columns have the row max
# Get which column has the max.
# Then, get this actual max...

# order = transition_matrix.loc[transition_matrix.idxmax(axis=0).values].max(axis=1).sort_values(ascending=False).unique().to_frame().index
order = transition_matrix.max(axis=1).sort_values(ascending=False).to_frame().index
# order = pd.Index(pd.Series(transition_matrix.loc 
#                            [
#                                transition_matrix.idxmax(axis=0).values # which column has the max? Order by that. 
#                            ] 
#                            .max(axis=1).sort_values(ascending=False).index.unique().tolist() 
#                            + 
#                            transition_matrix.max(axis=0).sort_values(ascending=False).to_frame().index.tolist()).unique())
mat = transition_matrix[order].reindex(order)

transition_matrix.to_csv('transition_matrix.csv')

transition_matrix.as_matrix()

Z = linkage(transition_matrix.as_matrix(), 'single', 'correlation')
# hcluster.dendrogram(Z, color_threshold=0)

fig = figure()
axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
D = transition_matrix.as_matrix()
Y = linkage(D, method='single', metric='correlation')
Z = dendrogram(Y, orientation='left')
axdendro.set_xticks([])
axdendro.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
index = Z['leaves']
D = D[index,:]
D = D[:,index]
im = axmatrix.matshow(mat, aspect='auto', origin='lower')
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
colorbar(im, cax=axcolor)

fig.show()



plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

plt.matshow(np.log(mat))

mat = mat.div(mat.sum(axis=1), axis=0)

fig = plt.figure(figsize=(10,10))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.PuBu
# cax = ax.matshow(transition_matrix[order].reindex(order),cmap=cmap)
cax = ax.matshow(transition_matrix[order].reindex(order),cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+order.tolist(),rotation=90)
ax.set_yticklabels(['']+order.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

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

mat2 = mat.drop(['M. Calcio', 'Primo Conti'])
mat2 = mat2.drop(['M. Calcio', 'Primo Conti'],axis=1)

fig = plt.figure(figsize=(10,10))#,dpi=300)
ax = fig.add_subplot(111)
cmap=plt.cm.PuBu
# cax = ax.matshow(transition_matrix[order].reindex(order),cmap=cmap)
cax = ax.matshow(mat2,cmap=cmap)
fig.colorbar(cax)

ax.set_xticklabels(['']+mat2.index.tolist(),rotation=90)
ax.set_yticklabels(['']+mat2.index.tolist())

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

# Create the actual edgelist for the transition matrix (of a first-order Markov chain)
df5 = df4.groupby(['from','to'])['total_people'].sum().to_frame()
df5.columns = ['weight']
df5.reset_index(inplace=True)
df5.head(10)

# Export
df5.to_csv('static_edgelist.csv')
nodes[['short_name','longitude','latitude']].to_csv('node_positions.csv')



