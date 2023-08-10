import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import random

get_ipython().magic('matplotlib inline')
import matplotlib as mpl
mpl.style.use("ggplot")
import matplotlib.colors as colors
import matplotlib.cm as cmx

import os
from operator import itemgetter

addresses = pd.read_csv("https://cs259c8bbcd0960x47fcxbe6.blob.core.windows.net/ppdata/Addresses.csv", low_memory=False)

companies = pd.read_csv("https://cs259c8bbcd0960x47fcxbe6.blob.core.windows.net/ppdata/Entities.csv", low_memory=False)

officers = pd.read_csv("https://cs259c8bbcd0960x47fcxbe6.blob.core.windows.net/ppdata/Officers.csv", low_memory=False)

inter = pd.read_csv("https://cs259c8bbcd0960x47fcxbe6.blob.core.windows.net/ppdata/Intermediaries.csv", low_memory=False)

edges = pd.read_csv("https://cs259c8bbcd0960x47fcxbe6.blob.core.windows.net/ppdata/all_edges.csv", low_memory=False)

G = nx.DiGraph()

for n,row in addresses.iterrows():
    G.add_node(row.node_id, node_type="address", details=row.to_dict())
    
for n,row in companies.iterrows():
    G.add_node(row.node_id, node_type="companies", details=row.to_dict())
    
for n,row in officers.iterrows():
    G.add_node(row.node_id, node_type="officers", details=row.to_dict()) 
    
for n,row in inter.iterrows():
    G.add_node(row.node_id, node_type="intermediaries", details=row.to_dict())

for n,row in edges.iterrows():
    G.add_edge(row.node_1, row.node_2, rel_type=row.rel_type, details={})

#get all connected subgraphs
subgraphs = [g for g in nx.connected_component_subgraphs(G.to_undirected())]

#sort by number of nodes in each
subgraphs = sorted(subgraphs, key=lambda x: x.number_of_nodes(), reverse=True)

print([s.number_of_nodes()for s in subgraphs[:100]])

#accessing node lables
def get_node_label(n):
    if n["node_type"] == "address":
        if pd.isnull(n["details"]["address"]):
            return ""
        return n["details"]["address"].replace(";", "\n")
    return n["details"]["name"]

node_types = [
    "address",
    "companies",
    "intermediaries",
    "officers"
]

def plot_graph(g, label_nodes=True, label_edges=False, figsize=(15,15)):
    
    #leveraging node attributes
    node_to_int = {k: node_types.index(k) for k in node_types}
    node_colours = [node_to_int[n[1]["node_type"]] for n in g.nodes(data=True)]
    node_labels = {k:get_node_label(v) for k,v in g.nodes(data=True)}
    
    #matplotlib setup
    cmap = plt.cm.rainbow
    cNorm  = colors.Normalize(vmin=0, vmax=len(node_to_int)+1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    
    plt.figure(figsize=figsize)
    
    #node positioninig for networkx draw
    pos = nx.spring_layout(g, iterations=100)
    
    #graph drawing with networkx
    nx.draw_networkx_nodes(g, pos, node_color=node_colours, cmap=cmap, vmin=0, vmax=len(node_to_int)+1)
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(), arrows=True)
    
    if label_nodes:
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=10, font_family='sans-serif')
        
    if label_edges:
        edge_labels = {(e[0], e[1]): e[2]["rel_type"] for e in g.edges(data=True)}
        nx.draw_networkx.edge_labels(g, pos, edge_labels)
        

plot_graph(subgraphs[74], figsize=(15,15))

plot_graph(subgraphs[69], figsize=(15,15))

#setting up node degree exploration
nodes = g.nodes()
g_degree = g.degree()
types = [g.node[n]["node_type"] for n in nodes]
degrees = [g_degree[n] for n in nodes]
names = [get_node_label(g.node[n]) for n in nodes]
node_degree = pd.DataFrame(data={"node_type":types, "degree":degrees, "name": names}, index=nodes)

#sorting by node degree(number of neighbouring nodes)
node_degree.sort_values("degree", ascending=False)[0:20]

#working with page rank
pr = nx.pagerank_scipy(g)
node_degree["page_rank"] = node_degree.index.map(lambda x: pr[x])
node_degree.sort_values("page_rank", ascending=False)[0:20]

eg = nx.ego_graph(g, 10165699, radius=1)
plot_graph(eg)

from mpl_toolkits.basemap import Basemap
m = Basemap(projection='robin',lon_0=0,resolution='l')
m.drawcountries(linewidth = 0.5)
m.fillcontinents(color='white',lake_color='white')
m.drawcoastlines(linewidth=0.5)
plt.figure(figsize=(15,15))

from IPython.display import Image
Image("https://cs259c8bbcd0960x47fcxbe6.blob.core.windows.net/ppdata/Graph%20on%20Map.png")



