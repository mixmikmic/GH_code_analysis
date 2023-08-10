import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir('/Users/Carla/Documents/Network Analysis/trade-networks')

#let's view the source data
from IPython.display import Image
Image("UNCTAD.png")

os.chdir('/Users/Carla/Documents/Network Analysis/trade-networks/Clusters')
dfedges = pd.read_csv("SITC0 [Edges].csv", sep = ",")
dfedges = dfedges[['Source', 'Target', 'weight']]
dfedges.head(20)
#dfedges.shape (1228,3)

#Below sample for SITC0 (Food)

# All goods
os.chdir('/Users/Carla/Documents/Network Analysis/trade-networks')
Image("All.png")

# SITC 0: Food
Image("SITC0.png")

# SITC 1: Beverages and Tobacco
Image("SITC1.png")

# SITC 2: Crude
Image("SITC2.png")

# SITC 3: Mineral Fuels
Image("SITC3.png")

#SITC 4: Animal and Vegetable Oils  
Image("SITC4.png")

#SITC 5: Chemicals  
Image("SITC5.png")

#SITC 6: Manufacturing
Image("SITC6.png")

#SITC 7: Machinery 
Image("SITC7.png")

#SITC 8: Miscellaneous Manufacturing Goods
Image("SITC8.png")

#SITC 9 :Commodities
Image("SITC9.png")

os.chdir('/Users/Carla/Documents/Network Analysis/trade-networks/Clusters')
dfedges = pd.read_csv("SITC0 [Edges].csv", sep = ",")
dfedges = dfedges[['Source', 'Target', 'weight']]
tuples = [tuple(x) for x in dfedges.values]

G=nx.Graph() 
G.add_weighted_edges_from(tuples)

#Import the community detection API
# Documentation: http://perso.crans.org/aynaud/communities/api.html
import community
partition = community.best_partition(G)

community = pd.DataFrame(partition.items(), columns=['country', 'cluster'])
community.sort_values(by='cluster')
community.to_csv('networks.csv', sep='\t', encoding='utf-8')

#compute centers and group by dictionary values
community.cluster.max()

community.cluster.value_counts()

#create a subgraph using the cluster groupings
#then compute for the betweenness centrality





G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_path([0,1,2,3])
H = G.subgraph([0,1,2])
H.edges()

G=nx.erdos_renyi_graph(100, 0.01)
dendo=community.generate_dendrogram(G)
for level in range(len(dendo) - 1) :
    print("partition at level", level, "is", community.partition_at_level(dendo, level))



