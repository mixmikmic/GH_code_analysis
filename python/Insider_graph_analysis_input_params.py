import pandas as pd

import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

# Load data: Subset of 'logon' data (only for logoff events, processed and saved previously)

gparams = pd.read_csv('./Data_Subset/graph_params_logoff.csv')

gparams.shape

gparams.head()

gparams.isnull().sum()

# G(V,E,W)

unodes = list(gparams['user'].unique())
pcnodes = list(gparams['pc'].unique())

weighted_edges = [(row['user'], row['pc'], row['activity_cnt']) for idx, row in gparams.iterrows()]


len(weighted_edges)



# Constructing a bipartite graph

B = nx.Graph()
B.add_nodes_from(unodes, bipartite = 0) # users as nodes
B.add_nodes_from(pcnodes, bipartite = 1) # PCs as nodes
B.add_weighted_edges_from(weighted_edges)


nx.is_connected(B)

nx.is_bipartite(B)

#pos = nx.draw_spectral(B)
# # default
# plt.figure(1)
# nx.draw(B,pos)
# smaller nodes and fonts
#plt.figure(2)
# nx.draw(B,pos,node_size=4,font_size=6) 
# larger figure size
plt.figure(figsize=(30,30)) 
nx.draw_networkx(B)
plt.show()

# c = bipartite.color(B)
# nx.set_node_attributes(B, 'bipartite', c)
plt.figure(figsize=(30,30)) 
nx.draw_networkx(B, node_size = 700, node_color = ['lightgreen', 'cyan'])
plt.show()

# Node degree of the graph
node_degrees = nx.degree(B)

type(node_degrees)

node_degrees

user_pc = pd.DataFrame(node_degrees.items(), columns=['user', 'pc_count'])

user_pc.head()

user_pc = user_pc[user_pc.user.isin(gparams.user)]

user_pc.shape

len(gparams['user'].unique())

user_pc.to_csv('./Data_Subset/user_pc_gdegree.csv', index = False)



