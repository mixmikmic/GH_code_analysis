import networkx as nx
import os
import ads as ads 
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms import bipartite as bi

os.environ["ADS_DEV_KEY"] = "kNUoTurJ5TXV9hsw9KQN1k8wH4U0D7Oy0CJoOvyw"

ads.config.token = 'ADS_DEV_KEY' 

#Search for papers 
# 50 most cited papers recent papers
papers1 = list(ads.SearchQuery(q="*",
                               sort="citation_count desc",
                               year="2012, 2013, 2014, 2015, 2016, 2017",
                               rows=50 ))

# find author names
a = []
for i in papers1:
    authors1 = i.author
    a.append(authors1)
author_names = a

# find the journals
j = []
for i in papers1:
    journals1 = i.pub
    j.append(journals1)
journals = j

# find year of publication
y = []
for i in papers1:
    year1 = i.year
    y.append(year1)
year = y

# find titles
t = []
for i in papers1:
    title1 = i.title
    t.append(title1)
title = t

# find keywords
k = []
for i in papers1:
    keyword1 = i.keyword
    k.append(keyword1)
keyword = k

# create an initial df
df = pd.DataFrame({'Author_Names' : author_names,
                   'Journal':journals,
                   'Year':year,
                   'Title':title,
                   'Keyword':keyword
  })

# Expand the df with melt
s1 = df.apply(lambda x: pd.Series(x['Author_Names']),axis=1).stack().reset_index(level=1, drop=True)
s1.name = 'Author_Name'
s2 = df.apply(lambda x: pd.Series(x['Title']),axis=1).stack().reset_index(level=1, drop=True)
s2.name = 'Title'
s3 = df.apply(lambda x: pd.Series(x['Keyword']),axis=1).stack().reset_index(level=1, drop=True)
s3.name = 'Keyword'
df_m = df.drop(['Author_Names','Title', 'Keyword'], axis=1).join(s1)
df_m = df_m.join(s2)
df_m = df_m.join(s3)

print df_m.shapeG.nodes(data=True)

df_m.to_csv('top_50.csv', sep=',', encoding='utf-8')

G=nx.from_pandas_dataframe(df_m, 'Journal', 'Author_Name', ['Title', 'Year', 'Keyword'])

# Weighted Projections/Clustering
# Largest most connected graph - 200 cut-off 
big_subg = [i for i in nx.connected_component_subgraphs(G) if len(i) > 200]
# Largest:
sg_largest = big_subg[0] # largest connected subgraph

# weighted_projections applied to subgraph to separate the two components
Journals,Author_Names = bi.sets(sg_largest)  # split into bipartites
j_proj_sg_largest = bi.weighted_projected_graph(sg_largest, Journals) 
a_proj_sg_largest = bi.weighted_projected_graph(sg_largest, Author_Names)

j = j_proj_sg_largest.edges(data=True) 
a = a_proj_sg_largest.edges(data=True)

# Island Method 
def trim(g):
    g_temp = nx.Graph()
    edge_bunch = [i for i in g.edges(data=True)]    
    g_temp.add_edges_from(edge_bunch)
    return g_temp
a_sg_island =  trim(a_proj_sg_largest)
j_sg_island = trim(j_proj_sg_largest)

#centrality
a_degree = nx.degree_centrality(a_sg_island)
j_degree = nx.degree_centrality(j_sg_island)

print a_sg_island.number_of_nodes()
print j_sg_island.number_of_nodes()

print len(a_degree) # 12 journals
a_degree

len(j_degree) # 7268 authors

get_ipython().magic('matplotlib inline')
nx.draw(a_sg_island,pos=nx.spring_layout(a_sg_island)) # use spring layout

nx.draw(j_sg_island,pos=nx.spring_layout(j_sg_island)) # use spring layout

